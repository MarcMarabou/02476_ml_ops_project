# -*- coding: utf-8 -*-
import click
import logging
import glob

from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from zipfile import ZipFile
from os import path, makedirs, system
from tfrecord.torch.dataset import TFRecordDataset


def extract_data(file_path: str) -> None:
    """
    Extracts data downloaded from kaggle

    Args
        file_path (str) : File to extract
    """
    print("Extracting data")
    with ZipFile(file_path, "r") as zip:
        zip.extractall(path.join("data", "raw", "flowers"))


def download_data(download_path: str) -> None:
    system(
        "kaggle competitions download "
        + "-c tpu-getting-started "
        + f'-p "{download_path}"'
    )


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


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
