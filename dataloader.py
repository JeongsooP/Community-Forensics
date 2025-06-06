### Dataloader for fake/real image classification

import torch
import pandas as pd
import numpy as np
import os
import PIL.Image
import random
import custom_transforms as ctrans
import math
import utils as ut

from torchvision import transforms
#from torchvision.transforms import v2 as transforms
from torch.utils.data.distributed import DistributedSampler
from custom_sampler import DistributedEvalSampler
from functools import partial
import datasets as ds
import io
import logging

class dataset_huggingface(torch.utils.data.Dataset):
    """
    Dataset for Community Forensics
    """
    def __init__(
            self,
            args,
            repo_id='OwensLab/CommunityForensics',
            split='Systematic+Manual',
            mode='train',
            cache_dir='',
            dtype=torch.float32,
    ):
        """
        args: Namespace of argument parser
        split: split of the dataset to use
        mode: 'train' or 'eval'
        cache_dir: directory to cache the dataset
        dtype: data type
        """
        super(dataset_huggingface).__init__()
        self.args = args
        self.repo_id = repo_id
        self.split = split
        self.mode = mode
        self.cache_dir = cache_dir
        self.dtype = dtype
        self.dataset = self.get_hf_dataset()
    
    def __getitem__(self, index):
        """
        Returns the image and label for the given index.
        """
        data = self.dataset[index]
        image_bytes = data['image_data']
        label = int(data['label'])
        generator_name = data['model_name']

        img = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")

        return img, label, generator_name

    def get_hf_dataset(self):
        """
        Returns the huggingface dataset object
        """
        hf_repo_id = self.repo_id
        if self.mode == 'train':
            shuffle=True
            shuffle_batch_size=3000
        elif self.mode == 'eval':
            shuffle=False

        hf_dataset = ds.load_dataset(hf_repo_id, split=self.split, cache_dir=self.cache_dir, num_proc=self.args.cpus_per_gpu)
        if shuffle:
            hf_dataset = hf_dataset.shuffle(seed=self.args.seed, writer_batch_size=shuffle_batch_size, )

        return hf_dataset

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.dataset)
        
class dataset_folder_based(torch.utils.data.Dataset):
    """
    Dataset for sourcing images from a directory; designed to be used with the huggingface datasets library.
    """
    def __init__(
            self,
            args,
            dir,
            labels="real:0,fake:1",
            logger: logging.Logger = None,
            dtype=torch.float32,
    ):
        """
        args: Namespace of argument parser
        dir: directory to index
        labels: labels for the dataset. Default: "real:0,fake:1" -- assigns integer label 0 to images under "real" and 1 to images under "fake".
        dtype: data type

        The directory must be formatted as follows:
        - <generator_or_dataset_name>
            ∟ <label -- "real" or "fake">
                ∟ <image_name>.{jpg,png,...}
        `dir` should point to the parent directory of the `generator_or_dataset_name` folders.
        """
        super(dataset_folder_based).__init__()
        self.args = args
        self.dir = dir
        self.labels = self.parse_labels(labels)
        assert len(self.labels) == 2, f"Labels must be in the format 'label1:int,label2:int'. It only supports two labels. Instead, it is: {labels}."

        self.logger = logger
        if self.logger is None:
            self.logger = ut.logger
        self.dtype = dtype
        self.df = self.get_index(dir)

    def __getitem__(self, index):
        """
        Returns the image and label for the given index.
        """
        img_path = self.df.iloc[index]['ImagePath']
        label = int(self.df.iloc[index]['Label'])
        generator_name = self.df.iloc[index]['GeneratorName']

        img = PIL.Image.open(img_path).convert("RGB")

        return img, label, generator_name

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.df)
    
    def parse_labels(self, labels):
        """
        Parses the labels string and returns a dictionary of labels.
        """
        labels_dict = {}
        for label in labels.split(','):
            label_name, label_value = label.split(':')
            labels_dict[label_name] = int(label_value)

        return labels_dict

    def get_label_int(self, label):
        """
        Returns the integer label for the given label name.
        """
        if label in self.labels:
            return self.labels[label]
        else:
            raise ValueError(f"Label {label} not found in labels: {self.labels}. Please check the labels.")

    def get_index(self, dir):
        """
        Check the `dir` for the index file. If it exists, load it. If not, index the directory and save the index file.
        """
        index_path = os.path.join(dir, 'index.csv')
        if os.path.exists(index_path):
            df = pd.read_csv(index_path)
            if self.args.rank == 0:
                self.logger.info(f"Loaded index file from {index_path}")
        else:
            if self.args.rank == 0:
                self.logger.info(f"Index file not found. Indexing the directory {dir}. This may take a while...")
            df = self.index_directory(dir)
        return df

    def index_directory(self, dir, report_every=1000):
        """
        Indexes the given directory and returns a dataframe with the image paths, labels, and generator names.
        The directory must be formatted as follows:
        - <generator_or_dataset_name>
            ∟ <label -- "real" or "fake">
                ∟ <image_name>.{jpg,png,...}
        `dir` should point to the parent directory of the `generator_or_dataset_name` folders.
        """  
        df = pd.DataFrame(columns=['ImagePath', 'Label', 'GeneratorName'])
        temp_dfs=[]
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif')):
                    # get the generator name and label from the directory structure
                    generator_name=os.path.basename(os.path.dirname(root))
                    label=os.path.basename(root) # should be "real" or "fake"
                    label_int=self.get_label_int(label)
                    # get the image path
                    image_path=os.path.join(root, file)
                    # append the image path, label, and generator name to the list
                    temp_dfs.append(pd.DataFrame([[image_path, label_int, generator_name]], columns=['ImagePath', 'Label', 'GeneratorName']))
                    if len(temp_dfs) % report_every == 0 and self.args.rank == 0:
                        print(f"\rIndexed {len(temp_dfs)} images...          ", end='', flush=True)
        df = pd.concat(temp_dfs, ignore_index=True)
        print("") # print a new line after the progress bar
        # sort the dataframe by generator name, label, and image name
        df = df.sort_values(by=['GeneratorName', 'Label', 'ImagePath'])
        df = df.reset_index(drop=True)
        # save the dataframe
        df.to_csv(os.path.join(dir, 'index.csv'), index=False)

        self.logger.info(f"Indexed the directory {dir} and saved the index file to {os.path.join(dir, 'index.csv')}")
        return df
    
    def limit_real_data(self, df, num_max_images):
        """
        Limits the real data to contain `num_max_images` total images by preserving the smallest datasets first.
        """
        new_df=pd.DataFrame()
        # get the number of images per dataset name
        real_df = df[df['Label'] == 0]
        fake_df = df[df['Label'] == 1]

        if len(real_df) <= num_max_images:
            self.logger.info(f"The size of real data: {len(real_df)} is less than or equal to the target size: {num_max_images}. No need to limit the real data. Note that the original model is trained with near 50/50 real/fake to avoid bias -- too much deviation from this may lead to unwanted detection bias.")
            return df

        dataset_counts = real_df['GeneratorName'].value_counts()
        # sort the dataset counts in descending order
        dataset_counts = dataset_counts.sort_values(ascending=True)
        smallest_sum=0
        smallest_idx=0
        num_not_appended_datasets=len(dataset_counts)

        while True:
            perModelLen = dataset_counts.iloc[smallest_idx]
            if (perModelLen * num_not_appended_datasets + smallest_sum) >= num_max_images: # reached data target
                perModelLen = math.ceil((num_max_images - smallest_sum) / num_not_appended_datasets) # number of images to sample from datasets not yet fully appended
                break
            elif smallest_idx == len(dataset_counts)-1:
                break # ran out of datasets; this is when size of all real data is less than num_max_images
            else: # continuously grow perModelLen with the next smallest dataset
                smallest_sum += dataset_counts.iloc[smallest_idx] # fully append the smallest dataset
                smallest_idx+=1
                num_not_appended_datasets-=1

        # sample the datasets
        for dataset_name in dataset_counts.index[smallest_idx:]:
            dataset_df = real_df[real_df['GeneratorName'] == dataset_name]
            if len(dataset_df) > perModelLen:
                dataset_df = dataset_df.sample(n=perModelLen, random_state=self.args.seed)
            new_df = pd.concat([new_df, dataset_df], ignore_index=True)
        
        # append the remaining datasets
        for dataset_name in dataset_counts.index[:smallest_idx]:
            dataset_df = real_df[real_df['GeneratorName'] == dataset_name]
            new_df = pd.concat([new_df, dataset_df], ignore_index=True)

        # report the proportions per dataset
        if self.args.rank == 0:
            pd.options.display.float_format = '{:.2f} %'.format
            self.logger.info(f"Max images per dataset limited to {perModelLen}. Affected datasets: {dataset_counts.index[smallest_idx:]}")
            # Update the dataset counts for reporting proportions
            dataset_counts = new_df['GeneratorName'].value_counts()
            dataset_counts = dataset_counts / dataset_counts.sum() * 100 # composition percentage
            self.logger.info(f"Dataset composition: \n{dataset_counts}")
        
        # append the fake data
        new_df = pd.concat([new_df, fake_df], ignore_index=True)

        return new_df

def determine_resize_crop_sizes(args):
    """
    Determine resize and crop sizes based on input size.
    """
    if args.input_size==224:
        resize_size=256
        crop_size=224
    elif args.input_size==384:
        resize_size=440
        crop_size=384
    return resize_size, crop_size

def get_transform(args, mode="train", dtype=torch.float32):
    norm_mean = [0.485, 0.456, 0.406] #imagenet norm
    norm_std = [0.229, 0.224, 0.225]
    resize_size, crop_size = determine_resize_crop_sizes(args)
    augment_list = []

    if mode=="train":
        augment_list.append(transforms.Resize(resize_size))

        # RandomStateAugmentation
        if args.rsa_ops != '':
            # parse rsa_ops and their num_ops
            # Default "rsa_ops" is "JPEGinMemory,RandomResizeWithRandomIntpl,RandomCrop,RandomHorizontalFlip,RandomVerticalFlip,RRCWithRandomIntpl,RandomRotation,RandomTranslate,RandomShear,RandomPadding,RandomCutout"
            augment_list.append(ctrans.RandomStateAugmentation(resize_size=resize_size, crop_size=crop_size, auglist=args.rsa_ops, min_augs=args.rsa_min_num_ops, max_augs=args.rsa_max_num_ops))

        augment_list.append(transforms.RandomCrop(crop_size))

        # basic augmentations
        augment_list.extend([
            ctrans.ToTensor_range(val_min=0, val_max=1),
            transforms.Normalize(mean=norm_mean, std=norm_std),
            transforms.ConvertImageDtype(dtype)
        ])
    elif mode=="val" or mode=="test":
        augment_list.append(transforms.Resize(resize_size))
        augment_list.extend([
            transforms.CenterCrop(crop_size),
            ctrans.ToTensor_range(val_min=0, val_max=1),
            transforms.Normalize(mean=norm_mean, std=norm_std),
            transforms.ConvertImageDtype(dtype),
        ])
    transform = transforms.Compose(augment_list)
    return transform
        
class SubsetWithTransform(torch.utils.data.Dataset):
    """
    Custom subset class which allows to customize transform for each subsets got from random_split()
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.subset_len = len(subset)
        self.transform = transform
        
    def __getitem__(self, index):
        img, lab, generator_name = self.subset[index]
        if self.transform:
            img = self.transform(img)
        return img, lab, generator_name

    def __len__(self):
        return self.subset_len

def set_seeds_for_data(seed=11997733):
    """
    Set seeds for Python, numpy, and pytorch. Used to split the dataset consistantly across DDP instances.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_seeds_for_worker(seed=11997733, id=0):
    """
    Set seeds for python, and numpy. Default seed=11997733.
    PyTorch seeding is handled by torch.Generator passed into the DataLoader
    """
    seed = seed % (2**31)
    random.seed(seed+id)
    np.random.seed(seed+id)

def worker_seed_reporter(id=None):
    """
    Debug: reports worker seeds
    """
    workerseed = torch.utils.data.get_worker_info().seed
    numwkr = torch.utils.data.get_worker_info().num_workers
    baseseed = torch.initial_seed()
    print(f"Worker id: {id+1}/{numwkr}, worker seed: {workerseed}, baseseed: {baseseed}, workerseed % (2**31): {workerseed % (2**31)}")

def set_seeds_and_report(report=True, id=0):
    """
    Debug: set seeds and report
    """
    workerseed = torch.utils.data.get_worker_info().seed
    set_seeds_for_worker(workerseed, id)
    if report:
        worker_seed_reporter(id)

def get_seedftn_and_generator(args, seed=None):
    """
    Get the seed function and generator for the dataloader.
    Args:
        args: Namespace of argument parser
        seed: seed for random number generation
    """
    rank = args.rank
    if seed is not None:
        seedftn = partial(set_seeds_and_report, False)
        seed_generator = torch.Generator(device='cpu')
        seed_generator.manual_seed(seed+rank)
    else:
        seedftn = None
        seed_generator = None
        seed = random.randint(0, 1000000000)
    
    return seedftn, seed_generator, seed

def get_train_dataloaders(
    args,
    huggingface_repo_id='',
    huggingface_split='Systematic+Manual',
    additional_data_path='',
    additional_data_label_format='real:0,fake:1',
    batch_size=128,
    num_workers=4,
    val_frac=0.01,
    logger: logging.Logger = None,
    seed=None,
):
    """
    Get train and validation dataloaders for the dataset.
    Args:
        args: Namespace of argument parser
        huggingface_repo_id: huggingface repo id for the dataset
        huggingface_split: split of the dataset to use
        additional_data_path: path to the folder containing the dataset
        batch_size: size of batch
        num_workers: number of subprocesses to spawn
        val_frac: fraction of data to use for validation (default: 0.01)
        seed: seed for random number generation
    """
    rank = args.rank
    world_size = args.world_size

    seedftn, seed_generator, seed = get_seedftn_and_generator(args, seed)
    if logger is None:
        logger = ut.logger
    
    hf_dataset=None
    if huggingface_repo_id != '':
        hf_dataset=dataset_huggingface(args, huggingface_repo_id, split=huggingface_split, mode='train', cache_dir=args.cache_dir, dtype=torch.float32)
    
    folder_dataset=None
    if additional_data_path != '':
        folder_dataset=dataset_folder_based(args, additional_data_path, additional_data_label_format, logger=logger, dtype=torch.float32)
        num_fake_images = len(folder_dataset.df[folder_dataset.df['Label'] == 1])
        if hf_dataset is not None and not args.dont_limit_real_data_to_fake: # limit real data to the length of fake data
            num_hf_fake_images = len(hf_dataset.dataset.filter(lambda x: x['label'] == 1, num_proc=num_workers))
            num_hf_real_images = len(hf_dataset.dataset) - num_hf_fake_images
            num_fake_images = num_fake_images + num_hf_fake_images
            #num_real_images = num_hf_real_images + len(folder_dataset.df[folder_dataset.df['Label'] == 0])

            folder_based_real_limit = num_fake_images - num_hf_real_images
            if folder_based_real_limit < 0:
                folder_based_real_limit = 0
            else:
                if rank == 0:
                    logger.info(f"Limiting folder-based real data to {folder_based_real_limit} images to match the number of fake images.")
                folder_dataset.df = folder_dataset.limit_real_data(folder_dataset.df, folder_based_real_limit)

    # merge two datasets
    if hf_dataset is not None and folder_dataset is not None:
        dataset_object = torch.utils.data.ConcatDataset([hf_dataset, folder_dataset])
    elif hf_dataset is not None:
        dataset_object = hf_dataset
    elif folder_dataset is not None:
        dataset_object = folder_dataset
    else:
        raise ValueError("No dataset provided. Please provide a huggingface repo id or a folder path.")
    
    set_seeds_for_data(seed) # Set same seeds for dataset split

    # Split the dataset into train and validation sets
    train_frac = 1 - val_frac
    if val_frac > 0:
        traindata_split, valdata_split = torch.utils.data.random_split(dataset_object, (train_frac, val_frac))
    else:
        traindata_split = dataset_object
        valdata_split = []
    
    set_seeds_for_data(seed+rank) # after dataset is split, use different seeds for augmentations and shuffling.
    
    # Get dataloaders
    traindata_split = SubsetWithTransform(traindata_split, transform=get_transform(args, mode='train', dtype=torch.float32))
    train_sampler = DistributedSampler(
        traindata_split, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False,
    )
    trainloader = torch.utils.data.DataLoader(
        traindata_split, batch_size=batch_size, pin_memory=True,
        shuffle=False, num_workers=num_workers, sampler=train_sampler,
        worker_init_fn=seedftn, generator=seed_generator
    )

    if len(valdata_split) > 0:
        valdata_split = SubsetWithTransform(valdata_split, transform=get_transform(args, mode='val', dtype=torch.float32))
        val_sampler = DistributedEvalSampler(
            valdata_split, num_replicas=world_size, rank=rank, shuffle=False,
        )
        valloader = torch.utils.data.DataLoader(
            valdata_split, batch_size=batch_size, pin_memory=True,
            shuffle=False, num_workers=num_workers, sampler=val_sampler,
            worker_init_fn=seedftn, generator=seed_generator
        )
    else:
        valloader = None

    if rank == 0:
        if huggingface_repo_id != '':
            logger.info(f"Loaded huggingface dataset from {huggingface_repo_id}. Split: {huggingface_split}.")
        if additional_data_path != '':
            logger.info(f"Loaded folder dataset from {additional_data_path}.")
        logger.info(f"Train/Val split: num_total: {len(dataset_object)}, num_train: {len(traindata_split)}, num_val: {len(valdata_split)} ")

    return trainloader, valloader

def get_test_dataloader(
    args,
    huggingface_repo_id='',
    huggingface_split='PublicEval',
    additional_data_path='',
    additional_data_label_format='real:0,fake:1',
    batch_size=128,
    num_workers=4,
    logger: logging.Logger = None,
    seed=None,
):
    """
    Get test dataloader for the dataset.
    Args:
        args: Namespace of argument parser
        huggingface_repo_id: huggingface repo id for the dataset
        huggingface_split: split of the dataset to use
        additional_data_path: path to the folder containing the dataset
        batch_size: size of batch
        num_workers: number of subprocesses to spawn
        seed: seed for random number generation
    """
    rank = args.rank
    world_size = args.world_size

    if logger is None:
        logger = ut.logger

    seedftn, seed_generator, seed = get_seedftn_and_generator(args, seed)

    hf_dataset=None
    if huggingface_repo_id != '':
        hf_dataset=dataset_huggingface(args, huggingface_repo_id, split=huggingface_split, mode='eval', cache_dir=args.cache_dir, dtype=torch.float32)
    
    folder_dataset=None
    if additional_data_path != '':
        folder_dataset=dataset_folder_based(args, additional_data_path, additional_data_label_format, logger=logger, dtype=torch.float32)

    # merge two datasets
    if hf_dataset is not None and folder_dataset is not None:
        dataset_object = torch.utils.data.ConcatDataset([hf_dataset, folder_dataset])
    elif hf_dataset is not None:
        dataset_object = hf_dataset
    elif folder_dataset is not None:
        dataset_object = folder_dataset
    else:
        raise ValueError("No dataset provided. Please provide a huggingface repo id or a folder path.")
    
    set_seeds_for_data(seed+rank)

    # Create dataset subset with eval transform
    dataset_object = SubsetWithTransform(dataset_object, transform=get_transform(args, mode='val', dtype=torch.float32))

    # Get dataloaders
    test_sampler = DistributedEvalSampler(
        dataset_object, num_replicas=world_size, rank=rank, shuffle=True,
    )
    testloader = torch.utils.data.DataLoader(
        dataset_object, batch_size=batch_size, pin_memory=True,
        shuffle=False, num_workers=num_workers, sampler=test_sampler,
        worker_init_fn=seedftn, generator=seed_generator
    )
    if rank == 0:
        if huggingface_repo_id != '':
            logger.info(f"Loaded huggingface dataset from {huggingface_repo_id}. Split: {huggingface_split}.")
        if additional_data_path != '':
            logger.info(f"Loaded folder dataset from {additional_data_path}.")
        logger.info(f"Test set size: {len(dataset_object)} ")
    return testloader