import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random

from model import CLIPVAD
from scipy import ndimage
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label, save_best_record
import xd_option
import os


def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss


def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss


def hard_clips(scores):
    scores_np = scores.cpu().detach().numpy()
    scores_mean = np.mean(scores_np, 1, keepdims=True)  # Calculate the average
    scores_bin = np.where(scores_np > scores_mean, 1.0, 0.0)
    # Erosion
    erosion_large = ndimage.binary_erosion(scores_bin, structure=np.ones((1, 8))).astype(scores_bin.dtype)
    erosion_small = ndimage.binary_erosion(scores_bin, structure=np.ones((1, 4))).astype(scores_bin.dtype)

    idx_region_inner = scores.new_tensor(erosion_small - erosion_large)
    scores_region_inner = scores * idx_region_inner

    return scores_region_inner


def IVCL(logits, text_features, visual_features, labels, lengths, device):
    labels_b = 1 - labels[:, 0].reshape(labels.shape[0])
    labels_b = labels_b.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    labels_m = labels / torch.sum(labels, dim=1, keepdim=True)
    labels_m = labels_m.to(device)

    scores_region_inner = hard_clips(logits)

    hard_fg_scores = torch.zeros(0).to(device)
    easy_bg_scores = torch.zeros(0).to(device)

    hard_fg_features_all = []
    easy_bg_features_all = []
    for i in range(logits.shape[0]):
        tmp_top, idx_top = torch.topk(scores_region_inner[i], k=int(logits.shape[1] / 64 + 1), largest=True)
        tmp_top = torch.mean(tmp_top).view(1)
        hard_fg_scores = torch.cat([hard_fg_scores, tmp_top], dim=0)

        idx_top = idx_top.unsqueeze(-1).expand(idx_top.shape[0], visual_features.shape[-1])
        hard_fg_features = torch.gather(visual_features[i], 0, idx_top).mean(0)
        hard_fg_features_all.append(hard_fg_features)

        tmp_bot, idx_bot = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=False)
        tmp_bot = torch.mean(tmp_bot).view(1)
        easy_bg_scores = torch.cat([easy_bg_scores, tmp_bot], dim=0)

        idx_bot = idx_bot.unsqueeze(-1).expand(idx_bot.shape[0], visual_features.shape[-1])
        easy_bg_features = torch.gather(visual_features[i, 0:lengths[i]], 0, idx_bot).mean(0)
        easy_bg_features_all.append(easy_bg_features)

    hard_fg_features_all = torch.stack(hard_fg_features_all)
    easy_bg_features_all = torch.stack(easy_bg_features_all)

    hard_fg_features_norm = hard_fg_features_all / hard_fg_features_all.norm(dim=-1, keepdim=True)
    easy_bg_features_norm = easy_bg_features_all / easy_bg_features_all.norm(dim=-1, keepdim=True)
    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    logits_fg = hard_fg_features_norm @ text_features_norm.t().type(hard_fg_features_norm.dtype) / 0.07
    logits_bg = easy_bg_features_norm @ text_features_norm.t().type(easy_bg_features_norm.dtype) / 0.07
    logits_m = torch.cat([logits_bg[labels_m.shape[0] // 2:], logits_fg[labels_m.shape[0] // 2:]], dim=0)
    clsloss_m = -torch.mean(torch.sum(labels_m * F.log_softmax(logits_m, dim=1), dim=1), dim=0)

    logits_2 = torch.cat([easy_bg_scores[labels_b.shape[0] // 2:], hard_fg_scores[labels_b.shape[0] // 2:]], dim=0)
    clsloss_2 = F.binary_cross_entropy(logits_2, labels_b)

    return clsloss_m + clsloss_2


def COSL(soft_features, hard_features):
    soft_features = soft_features / soft_features.norm(dim=-1, keepdim=True)
    hard_features = hard_features / hard_features.norm(dim=-1, keepdim=True)
    scores = F.cosine_similarity(soft_features, hard_features, dim=-1)
    loss = 1.0 - torch.mean(scores)
    return loss


def train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device):
    if not os.path.exists('model'):
        os.makedirs('model')

    if not os.path.exists('output'):
        os.makedirs('output')

    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_class, prompt_text = get_prompt_text(label_map, args.prompt_json)
    ap_best = 0
    epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch + 1, " ap:", ap_best)

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = 0
            normal_features, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_class, label_map).to(device)
            
            soft_features, hard_features, visual_features, logits1, logits2 = model(visual_features, None, prompt_class, prompt_text, feat_lengths)

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss_total1 += loss1.item()

            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()

            loss3 = COSL(soft_features, hard_features)

            loss4 = torch.zeros(1).to(device)
            text_feature_normal = soft_features[0] / soft_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, soft_features.shape[0]):
                text_feature_abr = soft_features[j] / soft_features[j].norm(dim=-1, keepdim=True)
                loss4 += torch.abs(text_feature_normal @ text_feature_abr)
            loss4 = loss4 / 6

            loss5 = IVCL(logits1, soft_features, visual_features, text_labels, feat_lengths, device)

            loss = loss1 + loss2 + loss3 * 4.0 + loss4 * 1e-1 + loss5 * 1e-1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * normal_loader.batch_size * 2
            if step % 2560 == 0 and step != 0:
                print('epoch: ', e + 1, '| step: ', step, '| loss1: ', loss_total1 / (i + 1), '| loss2: ',
                      loss_total2 / (i + 1), '| loss3: ', loss3.item(), '| loss4: ', loss4.item())
                AUC, AP, MAP = test(model, test_loader, args.visual_length, prompt_class, prompt_text, gt, gtsegments, gtlabels, device)

                if AP > ap_best:
                    ap_best = AP
                    checkpoint = {
                        'epoch': e + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}
                    torch.save(checkpoint, args.checkpoint_path)
                    torch.save(model.state_dict(), args.model_path)
                    save_best_record(e + 1, step, AUC, AP, MAP, os.path.join(args.output_path))

        scheduler.step()

        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot',
                      'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    normal_dataset = XDDataset(args.visual_length, args.train_list, False, label_map, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    
    anomaly_dataset = XDDataset(args.visual_length, args.train_list, False, label_map, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    test_dataset = XDDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head,
                    args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix,
                    args.gate_blend, device)
   
    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
