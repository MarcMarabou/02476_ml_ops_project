from glob import glob
from os import path, makedirs

from src.models.task import get_args
from src.models.ViT import ViT

import torch

def main():
    # Return arguements
    args = get_args()

    # Find the model filename
    model_path = glob(
        (f"models/trained_models/{args.id_to_script}/*.ckpt")
    )


    # Load in the model
    model = ViT.load_from_checkpoint(model_path[0])
    # Script the model
    file_dir = f"models/scripted_models/{args.id_to_script}/"
    if not path.exists(file_dir):
        makedirs(file_dir)

    model.to_torchscript(
        file_path=path.join(file_dir, "deployable_model.pt"),
        method="trace",
        example_inputs=torch.randn(1, 3, 224, 224)
    )


if __name__ == "__main__":
    main()
