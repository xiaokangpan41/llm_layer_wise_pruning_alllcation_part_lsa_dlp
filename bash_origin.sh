#!/bin/bash

# 1. 设置显卡 (0号卡)
export CUDA_VISIBLE_DEVICES=0

# 2. 设置模型路径
MODEL="/home/xiaokang/llama_hf/Llama-2-7b-hf"

# 3. 开启调试打印
set -x

echo "Starting Baseline Run (Fixed)..."

# 4. 运行 Python 脚本
# 关键修复：
# 1. 使用 --sparsity_ratio 0.7 (这是根据你 grep 结果确认的参数名)
# 2. 保留 --fp16 (防止 OOM)
# 3. 这里的 python 使用绝对路径

/home/xiaokang/miniconda3/envs/lsa/bin/python main.py \
    --base_model "$MODEL" \
    --sparsity_ratio 0.7 \
    --pruner wanda \
    --layer lsav1 \
    --num_examples 128 \
    --tasks wikitext \
    --fp16 \
    --combine_mode linear \
    --risk_alpha 0 \
    --Lamda 0 \
    --risk_k 1 \
    --risk_probe 0.1 \
    --risk_decay 1.0 \
    --probe_method magnitude \
    --block 0

    # 注意：最后一行结束后不要加反斜杠