# -*- coding: utf-8 -*-
import glob
import io
import logging
from os import makedirs, path, system
from pathlib import Path
from typing import List, Optional, Union
from zipfile import ZipFile

import click
import numpy as np
import tensorflow as tf
import torch
from dotenv import find_dotenv, load_dotenv
from PIL import Image
from tensorflow import compat
from tqdm import tqdm


def extract_data(file_path: str) -> None:
    """
    Extracts data downloaded from kaggle

    Args
        file_path : str
            File to extract
    """
    print("Extracting data")
    with ZipFile(file_path, "r") as zip:
        zip.extractall(path.join("data", "raw", "flowers"))


def download_data(download_path: str) -> None:
    """
    Downloads data from kaggle.
    Requires kaggle API key to be set up.

    Args:
        download_path : str
            Download destination
    """
    system(
        "kaggle competitions download "
        + "-c tpu-getting-started "
        + f'-p "{download_path}"'
    )


def process_tfrecord(
    data_path: str, output_path: str, sizes: Optional[List[str]] = ["224x224"]
) -> None:
    """
    Processes the flower data from TFRecord to normal PyTorch tensors.
    Serializes the resulting datasets and saves them.

    Args:
        data_path : str
            Path to folder that contains raw data
        output_path : str
            Path to folder where processed data should be stored
    """

    def parse_image(
        proto: tf.train.Example, split: str
    ) -> Union[tf.Tensor, tf.SparseTensor]:
        schema = {
            "id": tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string),
        }
        if split != "test":
            schema["class"] = tf.io.FixedLenFeature([], tf.int64)
        return tf.io.parse_single_example(proto, schema)

    for sz in sizes:
        print(f"Working on {sz} files...")
        files = {
            "train": glob.glob(path.join(data_path, f"*-{sz}/train/*.tfrec")),
            "val": glob.glob(path.join(data_path, f"*-{sz}/val/*.tfrec")),
            "test": glob.glob(path.join(data_path, f"*-{sz}/test/*.tfrec")),
        }

        for split in files:
            print(f"Parsing {split} TFRecords...")
            output_dir = path.join(output_path, split, sz)
            if not path.exists(output_dir):
                makedirs(output_dir)
            for i, file in enumerate(tqdm(files[split])):
                output = {}
                dataset = tf.data.TFRecordDataset(file).map(
                    lambda x: parse_image(x, split)
                )
                output["ids"] = [compat.as_text(f["id"].numpy()) for f in dataset]
                output["images"] = [
                    np.array(Image.open(io.BytesIO(f["image"].numpy())))
                    for f in dataset
                ]
                if split != "test":
                    output["classes"] = [f["class"].numpy() for f in dataset]
                torch.save(output, path.join(output_dir, f"{sz}-{i}.pt"))


@click.command()
@click.argument("input_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    flower_path = path.join(input_filepath, "flowers")
    if not path.exists(flower_path):
        makedirs(flower_path)
        zip_path = path.join(input_filepath, "tpu-getting-started.zip")
        if not path.exists(zip_path):
            download_data(input_filepath)
        extract_data(zip_path)
    output_dir = path.join(output_filepath, "flowers")
    process_tfrecord(flower_path, output_dir)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
