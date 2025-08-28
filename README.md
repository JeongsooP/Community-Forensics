Community Forensics: \
Using Thousands of Generators to Train Fake Image Detectors (CVPR 2025)
---

Repository for [Community Forensics: Using Thousands of Generators to Train Fake Image Detectors](https://arxiv.org/abs/2411.04125). \
([Project Page](https://jespark.net/projects/2024/community_forensics/), [Dataset (Full; 1.1TB)](https://huggingface.co/datasets/OwensLab/CommunityForensics), [Dataset (Small; 278GB)](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small))

## Description
This repository contains the training and evaluation pipeline for Community Forensics. The pipeline supports distributed data parallel through `torchrun` and accepts two data sources -- Hugging Face repo and local data. 
The two data sources can be used on their own or can be combined.

The training pipeline also contains the data augmentation technique used in our paper (`'RandomStateAugmentation'`), which is a modified version of `RandomAugmentation` that applies augmentation in random order, and in random numbers.

Training and evaluation results can be reported to `wandb` if `--wandb_token` argument is provided.

For a single-image evaluation-only pipeline, please check the [eval_single](https://github.com/JeongsooP/Community-Forensics/tree/eval_single) branch.

## Usage Examples

Install the requirements using `pip install -r requirements.txt`

**Model checkpoints are now available on Hugging Face: [384-input-version](https://huggingface.co/OwensLab/commfor-model-384), [224-input-version](https://huggingface.co/OwensLab/commfor-model-224)**. They can be loaded using the `--hf_model_repo` argument.
* Two additional libraries required: `huggingface-hub`, `transformers` (included in `requirements.txt`)

PyTorch Checkpoints for the pretrained models can also be downloaded [here (DropBox)](https://www.dropbox.com/scl/fi/e8titz35ci9a2ij1oq5mu/model_weights.tar?rlkey=tmyz3tjqf7b4dg071kypsgoal&st=09ud9hdj&dl=0). 


### Training with Hugging Face data only
By default, checkpoints will be saved in the directory of `models.py` script. You can override where the checkpoint will be saved by passing `--save_path` argument. This argument should follow the format: `/path/to/checkpoint/{model_name}.pt`. You can load the checkpoints by passing `--ckpt_path` argument.

Hugging Face data will be saved under the path specified by `--cache_dir` argument. Default path is `~/.cache`.

<details>
  <summary><b>Show example script</b></summary>

```
NUM_GPUS=1
NUM_CPUS_PER_GPU=4
BATCH_SIZE_PER_GPU=128

export OMP_NUM_THREADS=$NUM_CPUS_PER_GPU
export MASTER_ADDR=localhost
export MASTER_PORT=23949 # random port

torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --rdzv_id=123 --rdzv_backend=c10d train.py \
    --gpus $NUM_GPUS \
    --cpus-per-gpu $NUM_CPUS_PER_GPU \
    --epochs 20 \
    --batch_size $BATCH_SIZE_PER_GPU \
    --warmup_frac 0.2 \
    --use_amp \
    --huggingface_train_repo "OwensLab/CommunityForensics-Small" \
    --hf_split_train "train" \
    --verbose 2 \
```
</details>

### Training with Hugging Face data plus local data
In this case, the local files should be structured in the following way:
```
─ {root}
  └── {generator name or dataset name}
      └── {label}
          └── {image file}
```

Then, provide the path to `{root}` using the argument `--additional_train_data`. 
You may need to assign integer values to labels for training. This can be done through `--additional_data_label_format` argument. The default value is `real:0,fake:1`, which assigns `0` to images under `real` folder and `1` to `fake`.

If you wish to only train on local data, then you can pass `""` to the `--huggingface_train_repo` argument.

<details>
  <summary><b>Show example script</b></summary>

```
NUM_GPUS=1
NUM_CPUS_PER_GPU=4
BATCH_SIZE_PER_GPU=128

export OMP_NUM_THREADS=$NUM_CPUS_PER_GPU
export MASTER_ADDR=localhost
export MASTER_PORT=23949 # random port

torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --rdzv_id=123 --rdzv_backend=c10d train.py \
    --gpus $NUM_GPUS \
    --cpus-per-gpu $NUM_CPUS_PER_GPU \
    --epochs 20 \
    --batch_size $BATCH_SIZE_PER_GPU \
    --warmup_frac 0.2 \
    --use_amp \
    --huggingface_train_repo "OwensLab/CommunityForensics-Small" \
    --hf_split_train "train" \
    --additional_train_data "/path/to/additional_data/root" \
    --additional_data_label_format "real:0,fake:1" \
    --verbose 2 \
```
</details>

&nbsp;
Note that the local data doesn't necessarily have to contain both labels. For example, you can provide additional 'real' data when training using the full [Community Forensics dataset](https://huggingface.co/datasets/OwensLab/CommunityForensics):

<details>
  <summary><b>Show example script</b></summary>

Example local file structure:
```
─ {root}
  └── ImageNet
      └── real
          └── 000000.jpg
          └── ...
  └── LAION
      └── real
          └── 000000.png
          └── ...
  ...
```

```
NUM_GPUS=1
NUM_CPUS_PER_GPU=4
BATCH_SIZE_PER_GPU=128

export OMP_NUM_THREADS=$NUM_CPUS_PER_GPU
export MASTER_ADDR=localhost
export MASTER_PORT=23949 # random port

torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --rdzv_id=123 --rdzv_backend=c10d train.py \
    --gpus $NUM_GPUS \
    --cpus-per-gpu $NUM_CPUS_PER_GPU \
    --epochs 5 \
    --batch_size $BATCH_SIZE_PER_GPU \
    --warmup_frac 0.2 \
    --use_amp \
    --huggingface_train_repo "OwensLab/CommunityForensics" \
    --hf_split_train "Systematic+Manual" \
    --additional_train_data "/path/to/additional_data/root" \
    --additional_data_label_format "real:0,fake:1" \
    --verbose 2 \
```
</details>

### Evaluating Hugging Face data

<details>
  <summary><b>Show example script</b></summary>

```
NUM_GPUS=1
NUM_CPUS_PER_GPU=4
BATCH_SIZE_PER_GPU=128
HF_MODEL_REPO="OwensLab/commfor-model-384"
#CKPT_PATH="/path/to/checkpoint_file/commfor_train_ckpt.pt" # When using local PyTorch checkpoint. Use with `--ckpt_path $CKPT_PATH` argument

export OMP_NUM_THREADS=$NUM_CPUS_PER_GPU
export MASTER_ADDR=localhost
export MASTER_PORT=23949

torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --rdzv_id=123 --rdzv_backend=c10d eval.py \
    --gpus $NUM_GPUS \
    --cpus-per-gpu $NUM_CPUS_PER_GPU \
    --batch_size $BATCH_SIZE_PER_GPU \
    --hf_model_repo $HF_MODEL_REPO \
    --huggingface_test_repo "OwensLab/CommunityForensics" \
    --hf_split_test "PublicEval" \
    --verbose 2 \
```
</details>

### Evaluating Hugging Face data plus local data
Similar to training with local data, you can evaluate on test data by passing a structured data. You may assign the labels if it differs from the default setting -- `real:0,fake:1`.

<details>
  <summary><b>Show example script</b></summary>

Local files must be structured in the following way:

```
─ {root}
  └── {generator name or dataset name}
      └── {label}
          └── {image file}
```

```
NUM_GPUS=1
NUM_CPUS_PER_GPU=4
BATCH_SIZE_PER_GPU=128
HF_MODEL_REPO="OwensLab/commfor-model-384"
#CKPT_PATH="/path/to/checkpoint_file/commfor_train_ckpt.pt" # When using local PyTorch checkpoint. Use with `--ckpt_path $CKPT_PATH` argument

export OMP_NUM_THREADS=$NUM_CPUS_PER_GPU
export MASTER_ADDR=localhost
export MASTER_PORT=23949

torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --rdzv_id=123 --rdzv_backend=c10d eval.py \
    --gpus $NUM_GPUS \
    --cpus-per-gpu $NUM_CPUS_PER_GPU \
    --batch_size $BATCH_SIZE_PER_GPU \
    --hf_model_repo $HF_MODEL_REPO \
    --huggingface_test_repo "OwensLab/CommunityForensics" \
    --hf_split_test "PublicEval" \
    --additional_test_data "/path/to/additional_data/root" \
    --additional_data_label_format "real:0,fake:1" \
    --verbose 2 \
```
</details>

### Evaluating local data only
You can pass `""` for `--huggingface_test_repo` argument if you only want to evaluate on local data.

<details>
  <summary><b>Show example script</b></summary>

```
NUM_GPUS=1
NUM_CPUS_PER_GPU=4
BATCH_SIZE_PER_GPU=128
HF_MODEL_REPO="OwensLab/commfor-model-384"
#CKPT_PATH="/path/to/checkpoint_file/commfor_train_ckpt.pt" # When using local PyTorch checkpoint. Use with `--ckpt_path $CKPT_PATH` argument

export OMP_NUM_THREADS=$NUM_CPUS_PER_GPU
export MASTER_ADDR=localhost
export MASTER_PORT=23949

torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --rdzv_id=123 --rdzv_backend=c10d eval.py \
    --gpus $NUM_GPUS \
    --cpus-per-gpu $NUM_CPUS_PER_GPU \
    --batch_size $BATCH_SIZE_PER_GPU \
    --hf_model_repo $HF_MODEL_REPO \
    --huggingface_test_repo "" \
    --additional_test_data "/path/to/additional_data/root" \
    --additional_data_label_format "real:0,fake:1" \
    --verbose 2 \
```
</details>

## Arguments
<details>
  <summary><b>Show program arguments</b></summary>

`train.py` and `eval.py` use the same set of arguments:

```
usage: train.py [-h] [--gpus GPUS] [--gpus_list GPUS_LIST] [--cpus-per-gpu CPUS_PER_GPU] [--epochs EPOCHS] [--train_itrs TRAIN_ITRS] [--batch_size BATCH_SIZE] [--lr LR] [--weight_decay WEIGHT_DECAY]
                [--warmup_epochs WARMUP_EPOCHS] [--warmup_frac WARMUP_FRAC] [--no_lr_schedule] [--val_frac VAL_FRAC] [--test_frac TEST_FRAC] [--augment AUGMENT] [--num_ops NUM_OPS]
                [--ops_magnitude OPS_MAGNITUDE] [--rsa_ops RSA_OPS] [--rsa_min_num_ops RSA_MIN_NUM_OPS] [--rsa_max_num_ops RSA_MAX_NUM_OPS] [--eval_only] [--model_inner_dim MODEL_INNER_DIM]
                [--model_size MODEL_SIZE] [--input_size INPUT_SIZE] [--patch_size PATCH_SIZE] [--pretrained_path PRETRAINED_PATH] [--freeze_backbone] [--dont_add_sigmoid] [--use_amp]
                [--amp_dtype AMP_DTYPE] [--save_path SAVE_PATH] [--ckpt_save_path CKPT_SAVE_PATH] [--ckpt_path CKPT_PATH] [--ckpt_keep_count CKPT_KEEP_COUNT] [--only_load_model_weights]
                [--tokens_path TOKENS_PATH] [--wandb_token WANDB_TOKEN] [--cache_dir CACHE_DIR] [--dont_limit_real_data_to_fake] [--huggingface_train_repo HUGGINGFACE_TRAIN_REPO]
                [--huggingface_test_repo HUGGINGFACE_TEST_REPO] [--hf_split_train HF_SPLIT_TRAIN] [--hf_split_test HF_SPLIT_TEST] [--hf_model_repo HF_MODEL_REPO]
                [--additional_train_data ADDITIONAL_TRAIN_DATA] [--additional_test_data ADDITIONAL_TEST_DATA] [--additional_data_label_format ADDITIONAL_DATA_LABEL_FORMAT] [--verbose VERBOSE] [--seed SEED]
                [--debug_port DEBUG_PORT]

Train a binary classifier for fake image detection.

options:
  -h, --help            show this help message and exit
  --gpus GPUS           Number of GPUs
  --gpus_list GPUS_LIST
                        List of GPUs to use (comma separated). If set, overrides --gpus.
  --cpus-per-gpu CPUS_PER_GPU
                        Number of cpu threads per GPU
  --epochs EPOCHS       Number of epochs
  --train_itrs TRAIN_ITRS
                        Number of training iterations. If set, overrides --epochs.
  --batch_size BATCH_SIZE
                        Batch size
  --lr LR               Learning rate
  --weight_decay WEIGHT_DECAY
                        Weight decay
  --warmup_epochs WARMUP_EPOCHS
                        Warmup epochs. Can be fractions of an epoch.
  --warmup_frac WARMUP_FRAC
                        Set up a fraction of total iterations to be used as warmup. Overrides `--warmup_epochs`. (-1: disabled)
  --no_lr_schedule      If set, do not use lr scheduler
  --val_frac VAL_FRAC   Fraction of validation set
  --test_frac TEST_FRAC
                        Fraction of test set
  --augment AUGMENT     Augmentations to always use. Enter comma-separated string from the following:(singleJPEG, StochasticJPEG, rrc, flip, randaugment)
  --num_ops NUM_OPS     Number of operations
  --ops_magnitude OPS_MAGNITUDE
                        RandAugment magnitude (default=10), max=30
  --rsa_ops RSA_OPS     List of augmentations to use for RandomStateAugmentation. Provide a comma-separated list of augmentations to use for RSA
  --rsa_min_num_ops RSA_MIN_NUM_OPS
                        Minimum number of operations for each element in rsa_ops. Provide either a comma-separated list of integers or a single integer to be broadcasted to all elements.
  --rsa_max_num_ops RSA_MAX_NUM_OPS
                        Maximum number of operations for each element in rsa_ops. Provide either a comma-separated list of integers or a single integer to be broadcasted to all elements.
  --eval_only           If true, only evaluate model on test set.
  --model_inner_dim MODEL_INNER_DIM
                        Model inner dimension
  --model_size MODEL_SIZE
                        Model size. Small or tiny
  --input_size INPUT_SIZE
                        Input size. 224 or 384
  --patch_size PATCH_SIZE
                        Patch size for ViT models
  --pretrained_path PRETRAINED_PATH
                        Path to pretrained model
  --freeze_backbone     If set, freeze backbone of model
  --dont_add_sigmoid    If set, do not add sigmoid to model output when evaluating
  --use_amp             If set, use automatic mixed precision
  --amp_dtype AMP_DTYPE
                        Data type for automatic mixed precision. fp16 or bf16
  --save_path SAVE_PATH
                        Path to save model
  --ckpt_save_path CKPT_SAVE_PATH
                        Path to save model checkpoints and wandb. If empty, automatically determine from args.save_path.
  --ckpt_path CKPT_PATH
                        Path to load model checkpoint
  --ckpt_keep_count CKPT_KEEP_COUNT
                        Number of checkpoints to keep. If set to -1, keep all checkpoints.
  --only_load_model_weights
                        If set, only load weights from checkpoint path specified here. Does not load optimizer, scheduler, etc.
  --tokens_path TOKENS_PATH
                        Path containing all necessary tokens
  --wandb_token WANDB_TOKEN
                        Wandb token. If set, will use this token to login to wandb.
  --cache_dir CACHE_DIR
                        Path to cache hugging face dataset.
  --dont_limit_real_data_to_fake
                        If set, do not limit the size of real data to fake data.
  --huggingface_train_repo HUGGINGFACE_TRAIN_REPO
                        Hugging Face repo ID for the trainig dataset.
  --huggingface_test_repo HUGGINGFACE_TEST_REPO
                        Hugging Face repo ID for the test dataset.
  --hf_split_train HF_SPLIT_TRAIN
                        Hugging Face split for training data.
  --hf_split_test HF_SPLIT_TEST
                        Hugging Face split for test data.
  --hf_model_repo HF_MODEL_REPO
                        Hugging Face repository ID for the model. Note that `--ckpt_path` argument will override this argument.
  --additional_train_data ADDITIONAL_TRAIN_DATA
                        Path to additional data to use for training. The directory must follow a specific structure: <root>/<generator_name>/<real_or_fake>/<image_name>.<ext>. This flag should point to the
                        root directory of the additional data.
  --additional_test_data ADDITIONAL_TEST_DATA
                        Path to additional data to use for testing. The directory must follow a specific structure: <root>/<generator_name>/<real_or_fake>/<image_name>.<ext>. This flag should point to the
                        root directory of the additional data.
  --additional_data_label_format ADDITIONAL_DATA_LABEL_FORMAT
                        Format for additional data labels. The format should be a comma-separated list of key:value pairs, where key is the label and value is the corresponding integer value. For example,
                        'real:0,fake:1' means that images under 'real' directory will be labeled as 0 and images under 'fake' directory will be labeled as 1.
  --verbose VERBOSE     Verbosity.
  --seed SEED           Random seed
  --debug_port DEBUG_PORT
                        Debug port for Debugpy. If set, will wait for debugger to attach.
```
</details>


## Citation

```
@InProceedings{Park_2025_CVPR,
    author    = {Park, Jeongsoo and Owens, Andrew},
    title     = {Community Forensics: Using Thousands of Generators to Train Fake Image Detectors},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {8245-8257}
}
```
