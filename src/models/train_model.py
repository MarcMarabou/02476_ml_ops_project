import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
import wandb
from torch import nn, optim
from torchvision import datasets

from models.ViT import ViT

# initializes wandb
wandb.init(project="ml_ops_project", entity="ml_ops_team10")


class train(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>",
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=0.001)
        parser.add_argument("--batch_size", default=64)
        parser.add_argument("--save_model_to", default="trained_model.pt")
        parser.add_argument("--epochs", default=5)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        model = ViT()
        model.train()
        print(model)

        # Load the training data
        train_set, val_set = NotImplemented
        trainloader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True
        )

        # Define the loss function, optimizer and hyperparameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        epochs = args.epochs

        # Instiate the training losses
        train_losses = []
        val_losses = []

        # Training loop
        for e in range(epochs):
            running_loss = 0
            for images, targets in trainloader:

                # Clear the gradients, do this because gradients are accumulated
                optimizer.zero_grad()

                # Forward pass, then backward pass, then update weights
                output = model(images)
                loss = criterion(output, targets)
                loss.backward()

                # Take an update step and view the new weights
                optimizer.step()

                # Add the loss
                running_loss += loss.item()
            else:
                print(f"Training loss: {running_loss/len(trainloader)}")
                # Append the running_loss for each epoch
                train_losses.append(running_loss / len(trainloader))


if __name__ == "__main__":
    train()
