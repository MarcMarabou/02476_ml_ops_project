import argparse
import os
import sys
from datetime import datetime

import gcsfs
import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from src.data.FlowerDataset import FlowerDataset
from src.models.task import get_args
from src.models.ViT import ViT

class drift(Callback):
    def teardown(self, trainer, pl_module, stage):
        print("test")

def main():
    # Training settings
    args = get_args()

    # Load the training data
    predict_set = FlowerDataset(args.data_path, "224x224", "test")
    predictloader = torch.utils.data.DataLoader(
        predict_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = ViT.load_from_checkpoint(checkpoint_path=args.load_model_ckpt)
    trainer = Trainer(callbacks=[drift()])
    predictions = trainer.predict(model, predictloader)
    print(predictions)


if __name__ == "__main__":
    main()
