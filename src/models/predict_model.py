import argparse
from glob import glob
import os
import sys
from datetime import datetime

import gcsfs
import matplotlib.pyplot as plt
import torch

from copy import deepcopy

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from src.data.FlowerDataset import FlowerDataset
from src.models.task import get_args
from src.models.ViT import ViT

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

    filepath_model = glob(os.path.join('models/trained_models', args.load_timestamp, '*.ckpt'))

    model = ViT.load_from_checkpoint(checkpoint_path=filepath_model[0])
    trainer = Trainer()
    predictions = trainer.predict(model, predictloader)
    print(predictions)

    drift_detector = torch.load(os.path.join('models/trained_models', args.load_timestamp, '*.pt'))

    model_copy = deepcopy(model)
    model_copy.ViT[1] = torch.nn.Identity()

    feature_extractor = torch.nn.Sequential(
        model_copy,
        torch.nn.Flatten()
    )

    images = next(iter(predictloader))[0]

    features = feature_extractor(images)

    score = drift_detector(features)
    p_val = drift_detector.compute_p_value(features)
    print(f'Drift score {score:.2f} and p-value {p_val:.2f}')


if __name__ == "__main__":
    main()
