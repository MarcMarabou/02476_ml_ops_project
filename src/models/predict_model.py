import argparse
import os
import sys
from datetime import datetime

import gcsfs
import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Trainer

from src.data.FlowerDataset import FlowerDataset
from src.models.ViT import ViT
from src.models.task import get_args

def main():
    # Training settings
    args = get_args()

    # Load the training data
    test_set = FlowerDataset(args.data_path, "224x224", "test")
    testloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=len(test_set),
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = ViT(args=args)
    
    trainer.fit(model, trainloader, valloader)


if __name__ == "__main__":
    main()