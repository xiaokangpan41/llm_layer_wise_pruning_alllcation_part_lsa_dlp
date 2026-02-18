#!/bin/bash

# ================= 配置区域 =================
# 你的 Optuna 搜索脚本文件名
SEARCH_SCRIPT="optuna_search.py"

# 被调用的剪枝脚本 (main.py) 的路径
MAIN_SCRIPT="main.py"

# 【修改点】：这里改成你的本地模型绝对路径
MODEL="/home/xiaokang/llama_hf/Llama-2-7b-hf"

# 实验名称 (Optuna Study Name)
STUDY_NAME="llama2_7b_lsa_local_search"

# 显卡设置
export CUDA_VISIBLE_DEVICES=0
# ===========================================

echo "Starting Optuna Search: $STUDY_NAME using local model at $MODEL..."

# 运行 Optuna 搜索
# 注意：即使是本地路径，transformers 库也能自动识别
python $SEARCH_SCRIPT \
    --main_script "$MAIN_SCRIPT" \
    --base_model "$MODEL" \
    --study_name "$STUDY_NAME" \
    --storage "sqlite://${STUDY_NAME}.db" \
    --n_trials 50 \
    --timeout 7200 \
    \
    --final_s 0.7 \
    --pruner wanda \
    --layer lsa \
    --num_examples 128 \
    --tasks wikitext \
    \
    --fixed_arg "--block 128" \
    # --fixed_arg "--deepsparse" \
    # --fixed_arg "--onnx_export_path ./Llama-2-7B/chat-onnx"