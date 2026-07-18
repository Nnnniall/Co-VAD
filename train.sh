#!/bin/bash

mkdir -p logs

CUDA_VISIBLE_DEVICES=0 nohup python -u src/ucf_train.py > logs/ucf_train.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u src/xd_train.py > logs/xd_train.log 2>&1 &

echo "Training scripts launched in background."
