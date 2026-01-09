#!/bin/bash

# CLIP微调示例脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 数据集路径
DATASET_PATH="/root/autodl-tmp/dataset/OHD-Caps-train-sam3-cleaned"

# 输出目录
OUTPUT_DIR="/root/autodl-tmp/output/ft-clip"

# 预训练模型
MODEL_NAME="openai/clip-vit-large-patch14-336"

# 运行微调脚本
python ft_clip.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_seq_length=77 \
    --logging_steps 10 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 3 \
    --fp16 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --remove_unused_columns=False \
    --preprocessing_num_workers 8
