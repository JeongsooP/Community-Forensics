#!/bin/sh

NUM_GPUS=4
NUM_CPUS_PER_GPU=4
BATCH_SIZE_PER_GPU=128
# optionally pass wandb token as `--wandb_token=<your_token>` argument

export OMP_NUM_THREADS=$NUM_CPUS_PER_GPU
export MASTER_ADDR=localhost
export MASTER_PORT=29500

torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --rdzv_id=123 --rdzv_backend=c10d train.py \
    --gpus $NUM_GPUS \
    --cpus-per-gpu $NUM_CPUS_PER_GPU \
    --train_itrs 52000 \
    --batch_size $BATCH_SIZE_PER_GPU \
    --warmup_frac 0.2 \
    --use_amp \
    