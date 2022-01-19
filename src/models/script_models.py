from glob import glob
from os import path

from src.models.ViT import ViT
from src.models.task import get_args

def main():
    # Return arguements
    args = get_args()

    # Find the model filename
    model_path = glob(path.join(f"models/trained_models/{args.model_timestamp_to_script}/*.ckpt"))
    # Load in the model
    model = ViT.load_from_checkpoint(checkpoint_path=model_path)
    # Script the model
    model.to_torchscript(file_path=f"models/scripted_models/{args.model_timestamp_to_script}/deployable_model.pt")

if __name__ == "__main__":
    main()




