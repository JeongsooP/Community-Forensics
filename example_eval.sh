#!/bin/sh

NUM_GPUS=1
NUM_CPUS_PER_GPU=4
BATCH_SIZE_PER_GPU=128
#CKPT_PATH="path/to/your/checkpoint.pt" # Use with `--ckpt_path` argument when loading local checkpoint
HF_MODEL_REPO="OwensLab/commfor-model-384"
# optionally pass wandb token as `--wandb_token=<your_token>` argument

export OMP_NUM_THREADS=$NUM_CPUS_PER_GPU
export MASTER_ADDR=localhost
export MASTER_PORT=29500

torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --rdzv_id=123 --rdzv_backend=c10d eval.py \
    --gpus $NUM_GPUS \
    --cpus-per-gpu $NUM_CPUS_PER_GPU \
    --batch_size $BATCH_SIZE_PER_GPU \
    --hf_model_repo $HF_MODEL_REPO