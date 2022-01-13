import argparse
from datetime import datetime
import os
import sys

import matplotlib.pyplot as plt
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from torch import nn, optim
from torchvision import datasets

from src.data.FlowerDataset import FlowerDataset
from src.models.ViT import ViT


def get_args():
    """Argument parser.
    Returns:
        Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(
        description="Script for running training",
        usage="python train_model.py <command>",
    )
    # ===== HYPER PARAMETERS =====
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        metavar="N",
        help="Image size (Default: 224)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        metavar="N",
        help="Image patch size (Default: 16)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=104,
        metavar="N",
        help="Number of unique image classes (Default: 104)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=12,
        metavar="N",
        help="TODO: Help text (Default: 12)",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.0,
        metavar="R",
        help="Dropout rate during training (Default: 0.0)",
    )
    parser.add_argument(
        "--dropout-attn",
        type=float,
        default=0.0,
        metavar="R",
        help="TODO: Help text (Default: 0.0)",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=768,
        metavar="N",
        help="Embed dimension (Default: 768)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=12,
        metavar="N",
        help="ViT model depth (Default: 12)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (Default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (Default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        metavar="M",
        help="ADAM momentum (Default: 0.0)",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=1,
        metavar="N",
        help="minimum number of epochs to train (Default: 1)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
        metavar="N",
        help="maximum number of epochs to train (Default: 5)",
    )
    # ===== RUNTIME SETTINGS =====
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        metavar="N",
        help="number of GPUs to train on (Default: None)",
    )
    parser.add_argument(
        "--auto-select-gpus",
        action="store_true",
        help="pick available gpus automatically (Default: False)",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        metavar="PATH",
        help="The directory to store the model (Default: None)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        metavar="N",
        help="how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process (Default: 0)",
    )
    parser.add_argument(
        "--wandb-api-key",
        type=str,
        default=None,
        metavar="KEY",
        help="wandb API key for logging (Default: None)",
    )

    args = parser.parse_args()
    return args


def main():
    # Training settings
    args = get_args()

    logger = None
    if args.wandb_api_key:
        logger = pl_loggers.WandbLogger(
            name="ViT", version=datetime.now().strftime("%Y%m%d%H%M%S"),
            project="ml_ops_project", entity="ml_ops_team10", config=args
        )
        wandb.login(key=args.wandb_api_key)
        print("Using wandb for logging.")
    else:
        logger = pl_loggers.TensorBoardLogger(
            name="ViT", version=datetime.now().strftime("%Y%m%d%H%M%S"),
        )
        print("No wandb API key provided. Using local TensorBoard.")


    # Load the training data
    train_set = FlowerDataset("data/processed/flowers", "224x224", "train")
    val_set = FlowerDataset("data/processed/flowers", "224x224", "val")
    trainloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = ViT(args=args)
    trainer = Trainer(
        logger=logger,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        enable_checkpointing=True if args.model_dir else False,
        default_root_dir=args.model_dir,
        auto_select_gpus=args.auto_select_gpus,
        log_every_n_steps=2,
        gpus=args.gpus,
    )
    trainer.fit(model, trainloader, valloader)


if __name__ == "__main__":
    main()
