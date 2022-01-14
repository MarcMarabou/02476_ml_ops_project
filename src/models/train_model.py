import argparse
import os
import sys
from tokenize import Intnumber


import torch

from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.models.ViT import ViT
from src.data.FlowerDataset import FlowerDataset

def get_args():
    """Argument parser.
    Returns:
        Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(
        description='Script for running training',
        usage='python train_model.py <command>')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        metavar='N',
        help='input batch size for training (default: 64)')
    parser.add_argument(
        '--min-epochs',
        type=int,
        default=1,
        metavar='N',
        help='minimum number of epochs to train (default: 1)')
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=5,
        metavar='N',
        help='maximum number of epochs to train (default: 5)')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        metavar='LR',
        help='learning rate (default: 0.01)')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.0,
        metavar='M',
        help='ADAM momentum (default: 0.0)')
    parser.add_argument(
        '--gpus',
        type=int,
        default=0,
        help='number of GPUs to train on (default: None)')
    parser.add_argument(
        '--auto-select-gpus',
        default=False,
        help='pick available gpus automatically (default: False)')
    parser.add_argument(
        '--model-dir',
        default='models/',
        help='The directory to store the model (default: None)')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process (Default: 0)')
    parser.add_arguement(
        '--fast-dev-run',
        type=int,
        default=False,
        help='Runs n if set to n (int) of train, val and test to find any bugs (ie: a sort of unit test) (Default: False).'
    )

    args = parser.parse_args()
    return args
    
def main():
    # Training settings
    args = get_args()

    # Load the training data
    train_set = FlowerDataset("data/processed/flowers", "224x224", "train")
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_set = FlowerDataset("data/processed/flowers", "224x224", "val")
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_dir, monitor="val_loss", mode="min")
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min")

    model = ViT(args=args)
    trainer = Trainer(
        logger=pl_loggers.WandbLogger(project="ml_ops_project", entity="ml_ops_team10"),
        callbacks=[checkpoint_callback, early_stopping_callback],
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        auto_select_gpus=args.auto_select_gpus,
        log_every_n_steps=2,
        gpus=args.gpus)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()
