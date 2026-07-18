from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, class_feature, weight, tokenized_prompts, flag=False):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if flag:
            x = self.transformer(x)
        else:
            counter = 0
            outputs = self.transformer.resblocks([x, class_feature, weight, counter])
            x = outputs[0]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class CLIPVAD(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 gate_blend: float,
                 device):
        super().__init__()
        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.gate_alpha = gate_blend
        self.device = device

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj(device)
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.classifier = nn.Linear(visual_width, 1)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.meta_net = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width // 4, bias=True)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width // 4, 4 * visual_width, bias=True))
        ]))
        self.encode_text = TextEncoder(self.clipmodel)

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.text_features_cache = None

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x, _ = self.temporal((images, None))
        x = x.permute(1, 0, 2)

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1])
        x1_h = self.gelu(self.gc1(x, adj))
        x2_h = self.gelu(self.gc3(x, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)

        return x
    
    def soft_prompt(self, text, class_feature, weight):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.encode_text(text_embeddings, class_feature, weight, text_tokens)

        return text_features
    
    def hard_prompt(self, des_dict):
        if hasattr(self, 'text_features_cache') and self.text_features_cache is not None:
            return self.text_features_cache

        category_features = []
        for class_name, descriptions in des_dict.items():
            desc_features = []
            for desc in descriptions:
                prompt = f"a video from a CCTV camera of a {class_name}. {desc}"
                text_token = clip.tokenize(prompt).to(self.device)
                text_embedding = self.clipmodel.encode_token(text_token)
                text_embedding = self.clipmodel.encode_text(text_embedding, text_token)
                desc_features.append(text_embedding.squeeze(0))
            desc_features = torch.stack(desc_features, dim=0)
            category_features.append(desc_features)
        text_features = torch.stack(category_features, dim=0)

        self.text_features_cache = text_features

        return text_features
    
    def forward(self, visual, padding_mask, class_name, prompt_text, lengths):
        visual_features = self.encode_video(visual, padding_mask, lengths)  # [B, T, D]

        # Base temporal anomaly logits (used for MIL + mining)
        logits1 = self.classifier(visual_features + self.mlp1(visual_features))  # [B, T, 1]

        hard_prompt_features = self.hard_prompt(prompt_text)  # [C, N, D]
        hard_prompt_features_norm = hard_prompt_features / hard_prompt_features.norm(dim=-1, keepdim=True)

        # Center gate to [-1, 1]: high-score snippets are enhanced while low-score
        # snippets are mildly down-weighted. Detach logits1 so logits2-path losses do
        # not directly update logits1 branch.
        gate = 2.0 * torch.sigmoid(logits1.detach()) - 1.0  # [B, T, 1]
        visual_features_enhanced = visual_features + self.gate_alpha * (visual_features * gate)  # [B, T, D]
        visual_features_norm = visual_features_enhanced / visual_features_enhanced.norm(dim=-1, keepdim=True)

        # Detach again to avoid logits2-related gradients pushing logits1 through attention.
        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features_enhanced
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)

        scores = []
        for i in range(hard_prompt_features_norm.shape[1]):
            temp_logits = visual_attn.squeeze(1) @ hard_prompt_features_norm[:, i, :].t().type(visual_features_norm.dtype) / 0.07
            max_logits = torch.max(temp_logits, dim=-1).values
            sp = torch.mean(max_logits)
            scores.append(sp.item())
        
        scores_tensor = torch.tensor(scores, device=self.device)
        k = max(1, int(len(scores) * 0.1))
        _, topk_indices = torch.topk(scores_tensor, k)
        mask = torch.zeros_like(scores_tensor, dtype=torch.bool)
        mask[topk_indices] = True
        scores = scores_tensor[mask].unsqueeze(1).unsqueeze(1)
        selected_embeddings = hard_prompt_features[:, mask].mean(dim=1)

        selected_embeddings_norm = selected_embeddings / selected_embeddings.norm(dim=-1, keepdim=True)
        hard_meta_features_meta = self.meta_net(selected_embeddings_norm)
        hard_meta_features_meta = hard_meta_features_meta.reshape(hard_meta_features_meta.shape[0], -1, self.visual_width)
        soft_prompt_features = self.soft_prompt(class_name, hard_meta_features_meta, 1.0)

        visual_attn = visual_attn.expand(visual_attn.shape[0], soft_prompt_features.shape[0], visual_attn.shape[2])

        soft_text_features = soft_prompt_features.unsqueeze(0)
        soft_text_features = soft_text_features.expand(visual_attn.shape[0], soft_text_features.shape[1], soft_text_features.shape[2])
        soft_text_features = soft_text_features + visual_attn
        soft_text_features = soft_text_features + self.mlp2(soft_text_features)
        
        soft_text_features_norm = soft_text_features / soft_text_features.norm(dim=-1, keepdim=True)
        soft_text_features_norm = soft_text_features_norm.permute(0, 2, 1)
        logits2 = visual_features_norm @ soft_text_features_norm.type(visual_features_norm.dtype) / 0.07
        
        # Return enhanced visual features for downstream mining/losses (e.g., IVCL)
        return soft_prompt_features, selected_embeddings, visual_features_enhanced, logits1, logits2
