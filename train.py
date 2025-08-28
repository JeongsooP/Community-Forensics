import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import os, sys
import logging
import random
import utils as ut
import models
import wandb
import gc

from torch.nn.parallel import DistributedDataParallel as DDP

logger: logging.Logger = ut.logger

def train(
        args=None,
        logger=None,
):
    # initialize distributed training
    rank, local_rank, world_size = ut.dist_setup()

    # Set random seed
    torch.manual_seed(args.seed+rank)
    np.random.seed(args.seed+rank)
    random.seed(args.seed+rank)

    # Set device
    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size

    if rank==0 and args.wandb_token != "":
        ut.init_wandb(args)

    # Load data
    trainLoader, valLoader = ut.get_dataloader(args, mode='train')
    args = ut.get_epochs_for_itrs(args, len(trainLoader))
    trainLoaderLen = len(trainLoader)

    # Load model
    try:
        model = models.ViTClassifier(
            model_size=args.model_size,
            input_size=args.input_size,
            patch_size=args.patch_size,
            freeze_backbone=args.freeze_backbone,
            device=local_rank, dtype=torch.float32).to(args.local_rank)
    except Exception as e:
        logger.error(f"Error loading model. rank={rank}: {e}")
        sys.exit(1)

    # Set optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp) if args.use_amp else None
    #scaler = torch.GradScaler(device=local_rank, enabled=args.use_amp)

    # Set scheduler
    if args.warmup_frac > 0:
        warmup_steps=round(args.warmup_frac*ut.get_total_itrs(args, trainLoaderLen))
    else:
        warmup_steps = round(args.warmup_epochs * trainLoaderLen)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*trainLoaderLen-warmup_steps, eta_min=ut.get_min_lr(args)) # set min_lr = lr if args.no_lr_schedule.

    # Set loss function
    criterion = nn.BCEWithLogitsLoss()

    # Load checkpoint if set
    if args.ckpt_path != '': # note that if this is set, it overrides the `--hf_model_repo` argument.
        if args.only_load_model_weights:
            model = ut.load_only_weights(model, args.ckpt_path, rank)
            epoch_start = 0
            total_itr = 0
        else:
            model, optimizer, scheduler, epoch_start, total_itr = ut.load_checkpoint(model, optimizer, scheduler, scaler, args.ckpt_path, rank)
            epoch_start = epoch_start+1 # Since it saves current epoch for ckpt, not next.
    elif args.hf_model_repo != '':
        model = ut.load_ckpt_from_huggingface(model, args.hf_model_repo, rank)
        epoch_start = 0
        total_itr = 0
    else:
        epoch_start = 0
        total_itr = 0

    # try compiling the model
    try:
        model = torch.compile(model, dynamic=True)
    except Exception as e:
        logger.error(f"Error compiling model. rank={rank}: {e}")
        #sys.exit(1)

    # Set DistributedDataParallel
    model = DDP(model, device_ids=[local_rank]) #DDP
    torch.cuda.empty_cache()
    dist.barrier()
    logger.info(f"Model loaded and DDP set. rank={rank}")

    # Train
    local_window_loss=ut.LocalWindow(100)
    for epoch in range(epoch_start, args.epochs):
        gc.collect() # run garbage collection
        avgTrainLoss, total_itr = ut.train_one_epoch(
            args=args,
            epoch=epoch,
            model=model,
            train_loader=trainLoader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            local_window_loss=local_window_loss,
            warmup_steps=warmup_steps,
            rank=rank,
            itr=total_itr,
        )
        if valLoader is not None:
            valLoss, valAcc, valAP = ut.evaluate_one_epoch(
                args=args,
                epoch=epoch,
                model=model,
                dataloader=valLoader,
                criterion=criterion,
                rank=rank,
                evalName="Val",
                separate_eval=False,
                add_sigmoid=(not args.dont_add_sigmoid),
            )
            wandb_log_dict = {"epoch": epoch+1, "Loss/Train": avgTrainLoss, "Loss/Val": valLoss, "Acc/Val": valAcc, "AP/Val": valAP}
        else:
            valLoss, valAcc, valAP = -1, -1, -1
            wandb_log_dict = {"epoch": epoch+1, "Loss/Train": avgTrainLoss}
        scheduler.step()
        if rank<=0 and args.wandb_token != "":
            # log wandb
            wandb.log(
                wandb_log_dict, commit=False
            )
            wandb.finish()
        gc.collect() # run garbage collection

    # log wandb and save model
    if rank <= 0:
        torch.save(model.state_dict(), args.save_path)
        ut.save_checkpoint(model, optimizer, scheduler, scaler, epoch, total_itr, args.save_path.replace('.pt', '_ckpt.pt'))
        if args.ckpt_keep_count > 0:
            ut.keep_only_topn_checkpoints(args.save_path, args.ckpt_keep_count)

def main():
    args = ut.parse_args()
    args.random_port_offset = np.random.randint(-1000,1000) # randomize to avoid port conflict in same device
    
    if args.debug_port > 0:
        import debugpy
        debugpy.listen(('localhost', args.debug_port))
        logger.info(f"Waiting for debugger to attach on port {args.debug_port}...")
        debugpy.wait_for_client()
        debugpy.breakpoint()

    if args.gpus_list != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus_list
        logger.info(f"Setting CUDA_VISIBLE_DEVICES to {args.gpus_list}.")
        args.gpus = len(args.gpus_list.split(','))

    assert args.gpus <= torch.cuda.device_count(), f'Not enough GPUs! {torch.cuda.device_count()} available, {args.gpus} required.'
    assert args.gpus > 0, f'Number of GPUs must be greater than 0!'
    assert args.cpus_per_gpu > 0, f'Number of CPUs per GPU must be greater than 0!'

    if args.ckpt_save_path == '':
        args.ckpt_save_path = args.save_path

    logger.info(f"Spawning processes on {args.gpus} GPUs.")
    logger.info(f"Verbosity: {args.verbose} (0: None, 1: Every epoch, 2: Every iteration)")

    logger.info(f"Model save name: {os.path.basename(args.save_path)}")

    train(
        args=args,
        logger=logger,
    )

if __name__ == "__main__":
    main()