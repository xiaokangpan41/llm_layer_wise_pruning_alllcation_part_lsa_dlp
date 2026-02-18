#!/bin/bash

# --- [1. 环境配置] ---
# 设置显卡
export CUDA_VISIBLE_DEVICES=0

# 定义路径变量 (根据你的环境修改)
PYTHON_EXEC="/home/xiaokang/miniconda3/envs/lsa/bin/python"
MODEL_PATH="/home/xiaokang/llama_hf/Llama-2-7b-hf"
MAIN_SCRIPT="main.py"
OPTUNA_SCRIPT="optuna_search.py"  # 假设你上面的python代码保存为这个名字

# --- [2. 运行 Optuna 搜索] ---
echo "Starting Optuna Hyperparameter Search..."
echo "Model: $MODEL_PATH"

# 关键参数说明：
# --python: 指定 subprocess 调用的 python 解释器，保持环境一致
# --pruner wanda: 对应你手动运行的 pruner
# --layer lsa: 对应你手动运行的 layer
# --fixed_arg: 用于传递 optuna 脚本中未定义但 main.py 需要的参数 (如 --block 0)

$PYTHON_EXEC "$OPTUNA_SCRIPT" \
    --python "$PYTHON_EXEC" \
    --base_model "$MODEL_PATH" \
    --main_script "$MAIN_SCRIPT" \
    --study_name "wanda_lsa_search_v1" \
    --storage "sqlite:///optuna_wanda.db" \
    --n_trials 50 \
    --timeout 7200 \
    --final_s 0.7 \
    --pruner "wanda" \
    --layer "lsa" \
    --num_examples 128 \
    --tasks "wikitext" \
    --fp16 \
    --work_dir "optuna_results" \
    --fixed_arg "--block 0" 

# 注意：如果 main.py 需要更多固定参数，可以继续追加 --fixed_arg "..."