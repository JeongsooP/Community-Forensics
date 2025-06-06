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
(Will be updated!)

## Arguments

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
