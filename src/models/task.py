import argparse


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
        "--patch-size",
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
        "--load-model-ckpt",
        default=None,
        metavar="FILE",
        help="Filename of the model checkpoint (Default: None)"
    )
    parser.add_argument(
        "--model-timestamp-to-script",
        default=None,
        metavar="PATH",
        help="Name of the timestamp directory to load the .ckpt and make the scripted model (Default: None)"
    )
    parser.add_argument(
        "--data-path",
        default="data/processed/flowers",
        metavar="PATH",
        help='Path to data files (Default: "data/processed/flowers")',
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
    parser.add_argument(
        "--random-affine",
        type=bool,
        default=False,
        help="Use random affine transformation",
    )
    parser.add_argument(
        "--random-gauss",
        type=bool,
        default=False,
        help="Use random gaussian blur transformation",
    )
    parser.add_argument(
        "--random-hflip",
        type=bool,
        default=False,
        help="Use random horizontal flip transformation",
    )

    args = parser.parse_args()
    return args
