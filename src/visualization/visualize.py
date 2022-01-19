import matplotlib.pyplot as plt
import torch

from src.data.FlowerDataset import FlowerDataset


def rand100TrainingImages():
    data = FlowerDataset(
        flowers_path="data/processed/flowers", size="224x224", split="train"
    )
    trainloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

    plt.figure(figsize=(15, 15))
    for i in range(100):
        img = next(iter(trainloader))
        img = img[0][0, :, :, :].permute(1, 2, 0)
        plt.subplot(10, 10, i + 1)
        #    plt.subplots_adjust(wspace=0, hspace=0)
        plt.imshow(img)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("reports/figures/sampledTrainingImages.png")


rand100TrainingImages()
