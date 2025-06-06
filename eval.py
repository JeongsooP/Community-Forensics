import torch
import torch.nn as nn
import numpy as np
import sys
import logging
import random
import utils as ut
import models
import wandb

from torch.nn.parallel import DistributedDataParallel as DDP

logger: logging.Logger = ut.logger

def eval(
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
    testLoader = ut.get_dataloader(args, mode='test')

    # Load model
    try:
        model = models.ViTClassifier(args=args, device=local_rank, dtype=torch.float32).to(args.local_rank)
    except Exception as e:
        logger.error(f"Error loading model. rank={rank}: {e}")
        sys.exit(1)
    
    # Set loss function
    criterion = nn.BCEWithLogitsLoss()

    # Load checkpoint
    if args.ckpt_path != '':
        model = ut.load_only_weights(model, args.ckpt_path, rank)
        model.eval()
    else:
        raise ValueError("Checkpoint path is not set. Please provide a valid checkpoint path via `--ckpt_path` argument.")
    
    # Evaluate
    args.epochs = 1 # Set max epochs to 1 for evaluation; this is only for visual clarity.
    test_loss, test_acc, test_ap = ut.evaluate_one_epoch(
        args,
        epoch=0,
        model=model,
        dataloader=testLoader,
        criterion=criterion,
        rank=rank,
        evalName="test",
        separate_eval=True,
        add_sigmoid=(not args.dont_add_sigmoid),
    )

    # Log results
    if rank==0:
        if args.wandb_token != "":
            wandb.log({
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_ap": test_ap,
            })
            wandb.finish()
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test AP: {test_ap:.4f}")


def main():
    args = ut.parse_args()
    eval(
        args=args,
        logger=logger,
    )

if __name__ == "__main__":
    main()

