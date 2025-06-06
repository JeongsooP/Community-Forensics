## Utils for training pipeline

import torch.distributed as dist
import torch
import torchmetrics.classification as tmc
import os
import wandb
import numpy as np
import logging
import dataloader as dl
import pandas as pd
import time
import math
import argparse

def get_logger():
    """
    Get a logger instance with unbuffered output.
    """
    class UnbufferedStreamHandler(logging.StreamHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    logger = logging.getLogger("CommunityForensics")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s - %(levelname)s] - %(message)s"
    )
    ch = UnbufferedStreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

logger: logging.Logger = get_logger() # this will be used in all main scripts that loads this file

def parse_args():
    parser = argparse.ArgumentParser(description='Train a binary classifier for fake image detection.')

    # Pipeline arguments
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--gpus_list', type=str, default='', help='List of GPUs to use (comma separated). If set, overrides --gpus.')
    parser.add_argument('--cpus-per-gpu', type=int, default=4, help='Number of cpu threads per GPU')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--train_itrs', type=int, default=-1, help='Number of training iterations. If set, overrides --epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=float, default=3.0, help='Warmup epochs. Can be fractions of an epoch.')
    parser.add_argument('--warmup_frac', type=float, default=-1, help='Set up a fraction of total iterations to be used as warmup. Overrides `--warmup_epochs`. (-1: disabled)')
    parser.add_argument('--no_lr_schedule', action='store_true', help='If set, do not use lr scheduler')
    parser.add_argument('--val_frac', type=float, default=0.01, help='Fraction of validation set')
    parser.add_argument('--num_ops', type=int, default=2, help='Number of operations')
    parser.add_argument('--ops_magnitude', type=int, default=10, help="RandAugment magnitude (default=10), max=30")
    parser.add_argument('--rsa_ops', type=str, default="JPEGinMemory,RandomResizeWithRandomIntpl,RandomCrop,RandomHorizontalFlip,RandomVerticalFlip,RRCWithRandomIntpl,RandomRotation,RandomTranslate,RandomShear,RandomPadding,RandomCutout", help='List of augmentations to use for RandomStateAugmentation. Provide a comma-separated list of augmentations to use for RSA')
    parser.add_argument('--rsa_min_num_ops', type=str, default='0', help='Minimum number of operations for each element in rsa_ops. Provide either a comma-separated list of integers or a single integer to be broadcasted to all elements.')
    parser.add_argument('--rsa_max_num_ops', type=str, default='2', help='Maximum number of operations for each element in rsa_ops. Provide either a comma-separated list of integers or a single integer to be broadcasted to all elements.')

    # Model arguments
    parser.add_argument('--model_inner_dim', type=int, default=512, help='Model inner dimension')
    parser.add_argument('--model_size', type=str, default='small', help='Model size. Small or tiny')
    parser.add_argument('--input_size', type=int, default=384, help='Input size. 224 or 384')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size for ViT models')
    parser.add_argument('--freeze_backbone', action='store_true', help='If set, freeze backbone of model')
    parser.add_argument('--dont_add_sigmoid', action='store_true', help='If set, do not add sigmoid to model output when evaluating')
    parser.add_argument('--use_amp', action='store_true', help='If set, use automatic mixed precision')
    parser.add_argument('--amp_dtype', type=str, default='fp16', help='Data type for automatic mixed precision. fp16 or bf16')
    
    # path arguments (loading, saving, flags, etc)
    parser.add_argument('--save_path', type=str, default='', help='Path to save model')
    parser.add_argument('--ckpt_save_path', type=str, default='', help="Path to save model checkpoints and wandb. If empty, automatically determine from args.save_path.")
    parser.add_argument('--ckpt_path', type=str, default='', help='Path to load model checkpoint')
    parser.add_argument('--ckpt_keep_count', type=int, default=5, help='Number of checkpoints to keep. If set to -1, keep all checkpoints.')
    parser.add_argument('--only_load_model_weights', action='store_true', help='If set, only load weights from checkpoint path specified here. Does not load optimizer, scheduler, etc.')
    parser.add_argument("--tokens_path", type=str, default="", help="Path containing all necessary tokens")
    parser.add_argument("--wandb_token", type=str, default="", help="Wandb token. If set, will use this token to login to wandb.")
    parser.add_argument("--cache_dir", type=str, default="~/.cache", help="Path to cache hugging face dataset.")
    parser.add_argument("--dont_limit_real_data_to_fake", action="store_true", help="If set, do not limit the size of real data to fake data.")
    parser.add_argument("--huggingface_train_repo", type=str, default="OwensLab/CommunityForensics", help="Hugging Face repo ID for the trainig dataset.")
    parser.add_argument("--huggingface_test_repo", type=str, default="OwensLab/CommunityForensics", help="Hugging Face repo ID for the test dataset.")
    parser.add_argument("--hf_split_train", type=str, default="Systematic+Manual", help="Hugging Face split for training data.")
    parser.add_argument("--hf_split_test", type=str, default="PublicEval", help="Hugging Face split for test data.")
    parser.add_argument("--additional_train_data", type=str, default="", help="Path to additional data to use for training. The directory must follow a specific structure: <root>/<generator_name>/<real_or_fake>/<image_name>.<ext>. This flag should point to the root directory of the additional data.")
    parser.add_argument("--additional_test_data", type=str, default="", help="Path to additional data to use for testing. The directory must follow a specific structure: <root>/<generator_name>/<real_or_fake>/<image_name>.<ext>. This flag should point to the root directory of the additional data.")
    parser.add_argument("--additional_data_label_format", type=str, default="real:0,fake:1", help="Format for additional data labels. The format should be a comma-separated list of key:value pairs, where key is the label and value is the corresponding integer value. For example, 'real:0,fake:1' means that images under 'real' directory will be labeled as 0 and images under 'fake' directory will be labeled as 1.")

    # Misc arguments
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity. 0: no output, 1: per epoch output, 2: per iteration.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Determine "save_path" automatically if empty
    if args.save_path == "":
        args.save_path = f"./trained_model/{time.strftime('%Y%m%d-%H%M%S')}/commfor_train.pt"
    if args.ckpt_save_path == "":
        args.ckpt_save_path = os.path.join(os.path.dirname(args.save_path), "checkpoints")

    return args

def report_args(args, logger:logging.Logger):
    if args.use_amp:
        logger.info(f"AMP enabled.")
    if args.no_lr_schedule:
        logger.info(f"Will not use lr scheduler.")
    if args.ckpt_path != '':
        if args.only_load_model_weights:
            logger.info(f"Will only load model weights from {args.ckpt_path}.")
        else:
            logger.info(f"Will load model checkpoint from {args.ckpt_path}.")
    if not args.eval_only:
        if args.train_itrs > 0:
            logger.info(f"Training for {args.train_itrs} iterations.")
        else:
            logger.info(f"Training for {args.epochs} epochs.")
    else:
        logger.info(f"Will only evaluate model on test set.")
    if args.visualize_feature != '':
        logger.info(f"Will visualize feature of node: {args.visualize_feature}. This will override evaluation in the test phase.")
    if args.visualize_train:
        assert args.visualize_feature != '', 'visualize_feature must be set to visualize training data. Please set args.visualize_feature to appropriate node.'
        logger.info(f"Will visualize training data. (Note: will not visualize test data.)")
        if args.image_frac == -1 and args.image_per_model == -1:
            logger.warning(f"WARNING: using the entire training images per model for visualization. This may take a long time.")
        if args.num_models == -1:
            logger.warning(f"WARNING: using the entire training models for visualization. This may take a long time.")
    if args.eval_aug != '':
        logger.info(f"Using evaluation augmentations: {args.eval_aug}.")
    if args.eval_aug_str != '':
        logger.info(f"Using evaluation augmentation strengths: {args.eval_aug_str}.")
    if args.warmup_frac > 0:
        logger.info(f"Using {args.warmup_frac} fraction of total iterations as warmup.")
    else:
        logger.info(f"Using {args.warmup_epochs} epochs as warmup.")
    if args.freeze_backbone:
        logger.info(f"Freezing backbone of model.")
    if args.num_models != -1:
        logger.info(f"Using {args.num_models} models for training.")
    if args.image_frac != -1:
        logger.info(f"Using {args.image_frac} fraction of images for training.")
    if args.image_per_model != -1:
        logger.info(f"Using {args.image_per_model} images per model for training. Insufficient model ok: {args.insufficient_ok}, Suppress warning: {args.suppress_insufficient_warning}")
    if args.duplicate_data > 1:
        logger.info(f"Will duplicate data {args.duplicate_data} times during training.")
    if args.retain_proportion:
        logger.info(f"Retaining proportion of real image datasets in the training data.")
    if args.specified_proportion != '':
        logger.info(f"Specifying proportion as {args.specified_proportion}.")

class LocalWindow(): # local window averaging loss reporting object
    def __init__(self, maxsize):
        self.queue = []
        self.maxsize = maxsize
    
    def put(self, val, returnval=False):
        self.queue.append(val)
        if len(self.queue) > self.maxsize:
            self.queue.pop(0)
        if returnval:
            return self.calc_loss(always_report=True)
    
    def calc_loss(self, always_report=True):
        if not always_report:
            if len(self.queue) >= self.maxsize // 2 or len(self.queue) == self.maxsize:
                assert len(self.queue) > 0, "List cannot be length 0"
                avg = sum(self.queue) / len(self.queue)
                return avg
            else:
                return 0 # continue to accumulate
        else:
            assert len(self.queue) > 0, "List cannot be length 0"
            avg = sum(self.queue) / len(self.queue)
            return avg
        
def adjust_lr(optimizer, lr):
    """
    changes the learning rate of optimizer to lr
    """
    for g in optimizer.param_groups:
        g['lr'] = lr

def copy_lr(optim_src, optim_dst):
    """
    Copies learning rate from optim_src to optim_dst
    """
    src_lr = optim_src.param_groups[0]['lr']
    for dst_g in optim_dst.param_groups:
        dst_g['lr'] = src_lr

def dist_setup():
    """
    Setup for distributed data parallel using torchrun

    Example torchrun script:
    torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=localhost:29531 train.py {arguments}
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", init_method="env://")

    torch.cuda.set_device(local_rank)
    logger.info(f"Rank {rank} / Local rank {local_rank} / World size {world_size} intialized.")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return rank, local_rank, world_size

def dist_cleanup():
    dist.destroy_process_group()

def parse_floating_point_string(floating_point_string):
    """
    Parse a floating point string into torch dtype
    """
    if floating_point_string == "fp16":
        return torch.float16
    elif floating_point_string == "bf16":
        return torch.bfloat16
    else:
        raise ValueError(f"Invalid floating point string: {floating_point_string}. Must be 'fp16' or 'bf16'.")

def train_one_epoch(
        args,
        epoch,
        model,
        train_loader,
        optimizer,
        scheduler,
        criterion,
        scaler,
        local_window_loss,
        warmup_steps,
        rank,
        itr=0,
):
    model.train()
    running_loss=0.
    train_loader.sampler.set_epoch(epoch) # Set trainloader epoch
    device = model.device
    trainBM = Benchmarker(args, rank, benchmark_name="TrainModel", images_per_itr=args.batch_size)
    trainDLBM = Benchmarker(args, rank, benchmark_name="TrainDataLoad")

    for i, data in enumerate(train_loader, 0):
        trainDLBM.initialize()
        # Get inputs
        inputs, labels, _ = unpack_data_and_preprocess(data, device, torch.float32)
        dl_lat_pitr, _ = trainDLBM.end(report=True)

        # Zero the parameter gradients
        optimizer.zero_grad()

        itr += 1 # must return itr to track it across all epochs
        if itr < warmup_steps:
            adjust_lr(optimizer, args.lr * (itr+1) / warmup_steps)
        if args.train_itrs > 0 and itr > args.train_itrs:
            break

        # Forward + backward + optimize
        trainBM.start()
        with torch.autocast(device_type='cuda', dtype=parse_floating_point_string(args.amp_dtype), enabled=args.use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        local_loss = local_window_loss.put(loss.item(), returnval=True) # update local window loss
        running_loss += loss.item()
        scaler.scale(loss).backward() # AMP. scaler.scale(loss) will return loss if not enabled. If enabled, return scaled loss
        scaler.unscale_(optimizer) # AMP
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping
        scaler.step(optimizer) # AMP
        scaler.update() # AMP
        #optimizer.step()
        if itr > warmup_steps:
            scheduler.step()
        model_lat_pitr, model_thpt_pitr = trainBM.end(report=True)

        # Print statistics
        if args.verbose > 1 and rank == 0:
            try:
                print(f'\r[Train] Epoch {epoch+1}/{args.epochs} | Total itrs: {itr} / {args.train_itrs} | Iteration (this epoch) {i+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Local loss: {local_loss:.4f} | Latency(Model/DL): {model_lat_pitr:.2f}/{dl_lat_pitr:.2f} s | Throughput(Model): {model_thpt_pitr:.2f} imgs/s            ', end="", flush=True)
            except Exception as e:
                print(f"Error: {e}")
                print("Epoch", epoch)
                print("args.epoch", args.epochs)
                print("Iteration", i)
                print("maxitr", len(train_loader))
                print("Loss", loss.item())
                print("Local loss", local_loss)
        if rank==0 and args.wandb_token != "":
            wandb.log({
                "iteration": itr,
                "Loss (per itr)": loss.item(),
                "Local loss": local_loss,
                "Learning Rate": optimizer.param_groups[0]['lr'],
                "Latency (Model) (per itr)": model_lat_pitr,
                "Throughput (Model) (per itr)": model_thpt_pitr,
                "Latency (DataLoad) (per itr)": dl_lat_pitr,
            })
        
        trainDLBM.start()
        
    # Print statistics at the end of iteration
    model_lat, model_thpt = trainBM.compute(verbose=(rank==0))
    dl_lat, _ = trainDLBM.compute(verbose=(rank==0))
    if args.verbose > 0 and rank == 0:
        logger.info(f'\n[Train] Epoch {epoch+1}/{args.epochs} | Total itrs: {itr} / {args.train_itrs} | Iteration {i+1}/{len(train_loader)} | Loss: {loss.item():.4f} | Local loss: {local_loss:.4f} | Latency(Model/DL): {model_lat:.2f}/{dl_lat:.2f} s | Throughput(Model): {model_thpt:.2f} imgs/s     ')

    if rank == 0: # log wandb and save checkpoint
        if args.wandb_token != "":
            # Log wandb
            wandb.log(
                {"Loss (per epoch)": running_loss / len(train_loader),
                "Latency (Model)": model_lat,
                "Throughput (Model)": model_thpt,
                "Latency (DataLoad)": dl_lat,}, commit=False
            )
        # Save checkpoint
        ckpt_save_path = determine_ckpt_path(args, epoch)
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, itr, ckpt_save_path)

    avgTrainLoss = running_loss / len(train_loader)

    return avgTrainLoss, itr

def evaluate_one_epoch(
    args,
    epoch,
    model,
    dataloader,
    criterion,
    rank,
    evalName="Val",
    separate_eval=False,
    add_sigmoid=True,
):
    binaryAcc = tmc.BinaryAccuracy(dist_sync_on_step=False, process_group=None).to(rank)
    binaryAP = tmc.BinaryAveragePrecision(dist_sync_on_step=False, process_group=None).to(rank)

    if separate_eval:
        separate_evaluator = evalSeparately(args, evalName, epoch)

    model.eval()
    running_loss=0.
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs, labels, generator_names = unpack_data_and_preprocess(data, rank, torch.float32)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            if add_sigmoid:
                outputs = torch.sigmoid(outputs)

            if separate_eval:
                separate_evaluator.accumulate(outputs, labels, generator_names)

            # Calculate statistics for the entire test set
            binaryAcc.update(outputs, labels)
            binaryAP.update(outputs, labels.to(int))

            # Print statistics
            if args.verbose > 1 and rank == 0:
                print(f'\r[{evalName}] Epoch {epoch+1}/{args.epochs} | Iteration {i+1}/{len(dataloader)} | Loss: {loss.item():.4f}            ', end="", flush=True)
        print("") # print new line; avoid overwriting the last line

        if separate_eval:
            if add_sigmoid:
                if type(criterion) == type(torch.nn.BCEWithLogitsLoss()): # modify sigmoid-included loss to not-included loss
                    separate_criterion = torch.nn.BCELoss()
            else:
                separate_criterion = criterion
            separate_evaluator.calculate(separate_criterion, rank, logger)

        avgLoss = torch.tensor(running_loss / len(dataloader), device=rank, dtype=torch.float32)
        dist.all_reduce(avgLoss, op=dist.ReduceOp.AVG)
        avgLoss = avgLoss.item()

        binaryAccValue = binaryAcc.compute().item()
        binaryAPValue = binaryAP.compute().item()
    
    #avgLoss = running_loss / len(dataloader)
    if args.verbose > 0 and rank == 0:
        logger.info(f'\n[{evalName}] Epoch {epoch+1}/{args.epochs} | Iteration {i+1}/{len(dataloader)} | Loss: {avgLoss:.4f} | binaryAcc: {binaryAccValue:.4f} | binaryAP: {binaryAPValue:.4f}  ')

    return avgLoss, binaryAccValue, binaryAPValue

class evalSeparately():
    def __init__(
        self,
        args,
        evalName,
        epoch,
    ):
        self.args = args
        self.evalName = evalName
        self.epoch = epoch
        self.output_dict = dict()
        self.label_dict = dict()
        self.unique_generator_names = set() # get unique strings in generator_names
        self.accumulated_outputs=[]
        self.accumulated_labels=[]

    def accumulate(self, outputs, labels, generator_names):
        # accumulate outputs and labels separately for each string in generator_names
        # outputs and labels are assumed to be torch tensors, generator names are assumed to be strings

        # update unique_generator_names by apppending new generator names from generator_names
        self.unique_generator_names.update(set(generator_names))

        for generator_name in self.unique_generator_names:
            generator_idx = np.array(generator_names) == generator_name

            if self.output_dict.get(generator_name) is None:
                self.output_dict[generator_name] = outputs[generator_idx]
            else:
                self.output_dict[generator_name] = torch.cat([self.output_dict[generator_name], outputs[generator_idx]], dim=0)

            if self.label_dict.get(generator_name) is None:
                self.label_dict[generator_name] = labels[generator_idx]
            else:
                self.label_dict[generator_name] = torch.cat([self.label_dict[generator_name], labels[generator_idx]], dim=0)
    
        return
    
    def read_saved_data(self, generator_name):
        # read saved output/label data appropriately given generator_name and flags
        return self.output_dict[generator_name], self.label_dict[generator_name]

    def read_all_saved_data(self, generator_name_list):
        # read saved output/label data for all generator_name in generator_name_list and concatenate them into a single tensor
        if type(generator_name_list) != list: # if the list is not a list, then it is a single generator name
            return self.read_saved_data(generator_name_list) # if not list, fall back to read_saved_data

        for name in generator_name_list:
            output, label = self.read_saved_data(name)
            if name == generator_name_list[0]: # if first generator name, initialize outputs and labels
                outputs = output
                labels = label
            else: # if not first generator name, concatenate outputs and labels
                outputs = torch.cat([outputs, output], dim=0)
                labels = torch.cat([labels, label], dim=0)
        return outputs, labels

    @staticmethod
    def calculate_metrics(args, outputs, labels, criterion, rank, logger:logging.Logger, evalName="Val"):
        # now assume data comes with generator label
        binaryAcc = tmc.BinaryAccuracy(dist_sync_on_step=False, process_group=None).to(rank)
        binaryAP = tmc.BinaryAveragePrecision(dist_sync_on_step=False, process_group=None).to(rank)

        with torch.no_grad():
            loss = criterion(outputs, labels)

            # Calculate statistics
            binaryAcc.update(outputs, labels) # now use real-data-compensated accuracy.
            binaryAP.update(outputs, labels.to(int))

        # Print statistics at the end of iteration
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        loss = loss.item()
        binaryAccValue = binaryAcc.compute().item()
        binaryAPValue = binaryAP.compute().item()

        # Check if labels are homogeneous -- if so, AP calculation would not work properly. In such cases, return "None" for AP portion.
        if len(labels.unique()) == 1:
            binaryAPValue = -1
            if args.rank == 0:
                logger.warning(f"[{evalName}] Warning: Labels are homogeneous. Unique labels: {labels.unique()}. AP calculation will not work properly. Returning `-1` for AP value.")

        if args.verbose > 0 and rank == 0:
            logger.info(f'[{evalName}] Loss: {loss:.4f} | binaryAcc: {binaryAccValue:.4f} | binaryAP: {binaryAPValue:.4f}   ')

        return loss, binaryAccValue, binaryAPValue

    def calculate(self, criterion, rank, logger):
        # calculate metrics
        data = {}
        data['Loss']=[]
        data['Acc']=[]
        data['AP']=[]
        meanLoss=[]
        meanAcc=[]
        meanAP=[]

        for generator_name in sorted(self.unique_generator_names):
            output_all, label_all = self.read_saved_data(generator_name)
            loss, acc, ap = self.calculate_metrics(self.args, output_all, label_all, criterion, rank, logger, evalName=generator_name)
            if rank == 0:
                # log generator_name in wandb
                data['Loss'].append([f"Loss/{generator_name}", loss])
                data['Acc'].append([f"Acc/{generator_name}", acc])
                data['AP'].append([f"AP/{generator_name}", ap])
                # append to compute mean
                meanLoss.append(loss)
                meanAcc.append(acc)
                meanAP.append(ap)
                
        # Calculate total metrics
        if rank == 0:
            # Compute mean metrics
            meanLoss = sum(meanLoss) / len(meanLoss)
            meanAcc = sum(meanAcc) / len(meanAcc)
            meanAP = sum(meanAP) / len(meanAP)
            data['Loss'].append([f"MeanLoss/{self.evalName}", meanLoss])
            data['Acc'].append([f"MeanAcc/{self.evalName}", meanAcc])
            data['AP'].append([f"MeanAP/{self.evalName}", meanAP])
            
            if self.args.verbose > 0:
                logger.info(f'[Mean] MeanLoss: {meanLoss:.4f} | MeanAcc: {meanAcc:.4f} | MeanAP: {meanAP:.4f}            ')
                
            if self.args.wandb_token != "":
                table_loss = wandb.Table(data=data['Loss'], columns=["label", "value"])
                table_acc = wandb.Table(data=data['Acc'], columns=["label", "value"])
                table_ap = wandb.Table(data=data['AP'], columns=["label", "value"])
                wandb.log(
                    {
                        f"Loss/separate_{self.evalName}": wandb.plot.bar(table_loss, "label", "value", title=f"Loss/separate_{self.evalName}"), 
                        f"Acc/separate_{self.evalName}": wandb.plot.bar(table_acc, "label", "value", title=f"Acc/separate_{self.evalName}"), 
                        f"AP/separate_{self.evalName}": wandb.plot.bar(table_ap, "label", "value", title=f"AP/separate_{self.evalName}"), 
                        "AP/Test_Mean": meanAP,
                        "Acc/Test_Mean": meanAcc,
                        "Loss/Test_Mean": meanLoss,
                    },
                    commit=False
                )
                wandb.log( # also log table
                    {f"Loss/separate_{self.evalName}_table": table_loss,
                    f"Acc/separate_{self.evalName}_table": table_acc,
                    f"AP/separate_{self.evalName}_table": table_ap,
                    },
                    commit=False
                )
        return
        
def get_dataloader(args, mode="train"):
    # Retrieves appropriate dataloader automatically based on arguments
    if mode=="train":
        trainLoader, valLoader = dl.get_train_dataloaders(
            args,
            huggingface_repo_id=args.huggingface_train_repo,
            huggingface_split=args.hf_split_train,
            additional_data_path=args.additional_train_data,
            additional_data_label_format=args.additional_data_label_format,
            batch_size=args.batch_size,
            num_workers=args.cpus_per_gpu,
            val_frac=args.val_frac,
            logger=logger,
            seed=args.seed,
        )
        return trainLoader, valLoader
    elif mode=="test":
        testLoader = dl.get_test_dataloader(
            args,
            huggingface_repo_id=args.huggingface_test_repo,
            huggingface_split=args.hf_split_test,
            additional_data_path=args.additional_test_data,
            additional_data_label_format=args.additional_data_label_format,
            batch_size=args.batch_size,
            num_workers=args.cpus_per_gpu,
            logger=logger,
            seed=args.seed,
        )
        return testLoader
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'test'.")

def get_min_lr(args):
    # get minimum learning rate for scheduler
    if args.no_lr_schedule:
        return args.lr
    else:
        return args.lr * 0.001

def unpack_data_and_preprocess(data, device, dtype=torch.float32):
    inputs, labels, generator_names = data
    inputs = inputs.to(device)
    labels = labels.unsqueeze(dim=1).to(device, dtype=dtype)
    return inputs, labels, generator_names

def determine_ckpt_path(args, epoch):
    """
    Automatically determines checkpoint save path
    """
    ckpt_save_path = os.path.join(args.ckpt_save_path, f"checkpoint_{epoch}.pt")
    os.makedirs(args.ckpt_save_path, exist_ok=True)
    return ckpt_save_path

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, itr, ckpt_path):
    """
    Saves checkpoint to ckpt_path
    """
    if hasattr(model.module, "_orig_mod"):
        model_state_dict = model.module._orig_mod.state_dict()
    else:
        model_state_dict = model.module.state_dict()

    checkpoint = {
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
        'itr': itr,
    }
    torch.save(checkpoint, ckpt_path)

def keep_only_topn_checkpoints(ckpt_path, top_n=5):
    """
    Automatically keeps only top_n checkpoints in ckpt_path
    """
    ckpt_dir = os.path.dirname(ckpt_path)
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
    ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True) # sort by epoch, descending
    # remove all but most recent top_n checkpoints
    if len(ckpt_files) > top_n:
        for f in ckpt_files[top_n:]:
            os.remove(os.path.join(ckpt_dir, f))

    return

def load_checkpoint(model, optimizer, scheduler, scaler, ckpt_path, rank):
    """
    Loads checkpoint from ckpt_path
    """
    dist.barrier()
    map_location = {'cuda:0': f'cuda:{rank}'}
    #map_location = f'cuda:{rank}'
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    # clean checkpoint
    state_dict = checkpoint['model']
    if list(state_dict.keys())[0].startswith("_orig_mod"):
        checkpoint['model'] = {key.replace("_orig_mod.", ""): state_dict[key] for key in state_dict.keys()}
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    if 'scalar' in checkpoint.keys():
        scaler.load_state_dict(checkpoint['scaler'])
    epoch = checkpoint['epoch']
    itr = checkpoint['itr']

    if rank==0:
        print(f"Checkpoint loaded from {ckpt_path}. Epoch: {epoch}, Itr: {itr}")

    return model, optimizer, scheduler, epoch, itr

def load_only_weights(model, ckpt_path, rank):
    """
    Only loads the weights of the model from `ckpt_path`. Useful for warm-starting the model.
    """
    dist.barrier()
    map_location = {'cuda:0': f'cuda:{rank}'}
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    state_dict = checkpoint['model']
    if list(state_dict.keys())[0].startswith("_orig_mod"):
        checkpoint['model'] = {key.replace("_orig_mod.", ""): state_dict[key] for key in state_dict.keys()}
    model.load_state_dict(checkpoint['model'])
    if rank==0:
        print(f"Model weights loaded from {ckpt_path}")
    return model

def get_token(args, label):
    """
    Reads token from args.tokens_path and returns corresponding token
    """
    df = pd.read_csv(args.tokens_path)
    token = df[df['label'] == label]["token"].values[0]
    return token

def init_wandb(args):
    """
    Initializes wandb

    Inputs:
        args: parsed args using argparse
    """
    modelsavepath = os.path.dirname(args.save_path)
    modelgivenname = ".".join(os.path.basename(args.save_path).split('.')[:-1]) if os.path.basename(args.save_path).find('.') != -1 else os.path.basename(args.save_path)
    wandbname = os.path.join("wandb", modelgivenname)
    wandbpath = os.path.join(modelsavepath, wandbname)
    os.makedirs(wandbpath, exist_ok=True)
    if args.wandb_token != "":
        wandb_key = args.wandb_token
    else:
        wandb_key = get_token(args, "wandb")
    wandb.login(
        anonymous="never",
        key=wandb_key
    )
    wandb.init(
        project="community-forensics",
        config=args,
        save_code=True,
        dir=wandbpath,
        name=modelgivenname,
    )
    wandb.define_metric("iteration")
    wandb.define_metric("epoch")
    wandb.define_metric("Local loss", step_metric="iteration")
    wandb.define_metric("Loss (per itr)", step_metric="iteration")
    wandb.define_metric("Learning Rate", step_metric="iteration")
    wandb.define_metric("Loss (per epoch)", step_metric="epoch")
    wandb.define_metric("Val Acc", step_metric="epoch")
    wandb.define_metric("Val Loss", step_metric="epoch")
    wandb.define_metric("Train Epoch Time", step_metric="epoch")
    wandb.define_metric("Val Epoch Time", step_metric="epoch")
    wandb.define_metric("Data Load Epoch Time", step_metric="epoch")
    wandb.define_metric("Latency (Model) (per itr)", step_metric="iteration")
    wandb.define_metric("Throughput (Model) (per itr)", step_metric="iteration")
    wandb.define_metric("Latency (DataLoad) (per itr)", step_metric="iteration")
    wandb.define_metric("Throughput (DataLoad) (per itr)", step_metric="iteration")
    wandb.define_metric("Latency (Model)", step_metric="epoch")
    wandb.define_metric("Throughput (Model)", step_metric="epoch")
    wandb.define_metric("Latency (DataLoad)", step_metric="epoch")
    wandb.define_metric("Throughput (DataLoad)", step_metric="epoch")
    
    return

class Benchmarker():
    """
    Benchmark time for each epoch
    """
    def __init__(
        self,
        args,
        rank,
        benchmark_name="", # name of benchmark
        images_per_itr=None, # number of images per itr
    ):
        self.args = args
        self.rank = rank
        assert benchmark_name != "", "Name for which metric to benchmark must be provided."
        self.benchmark_name = benchmark_name
        self.images_per_itr = images_per_itr
        self.total_elapsed=0
        self.total_counts=0
        self.start_time=-1
        self.end_time=-1

    def initialize(self): # initialize benchmark for dataloader.
        if self.start_time < 0:
            self.start()

    def start(self):
        self.start_time = time.time()
    
    def end(self, report=False):
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        self.total_elapsed += elapsed_time
        self.total_counts += 1
        if report:
            current_throughput = self.images_per_itr / elapsed_time if self.images_per_itr is not None else None
            return elapsed_time, current_throughput
    
    def compute(self, verbose=True):
        avg_elapsed = self.total_elapsed / self.total_counts
        avg_throughput = self.images_per_itr / avg_elapsed if self.images_per_itr is not None else None
        if verbose:
            output_str = f"[{self.benchmark_name}] Total elapsed: {self.total_elapsed:.2f} s. Avg elapsed: {avg_elapsed:.2f} s."
            if avg_throughput is not None:
                output_str += f" Avg throughput: {avg_throughput:.2f} img/s. "
            print(output_str)
        return avg_elapsed, avg_throughput

    def reset(self):
        self.total_elapsed=0
        self.total_counts=0
        self.start_time=-1
        self.end_time=-1

def get_epochs_for_itrs(args, trainLoaderLen):
    # Returns appropriate number of epochs given itrs
    if args.train_itrs > 0:
        args.epochs = math.ceil(args.train_itrs / trainLoaderLen)
    return args

def get_total_itrs(args, trainLoaderLen):
    # Get total number of iterations
    if args.train_itrs > 0:
        return args.train_itrs
    else:
        return args.epochs * trainLoaderLen