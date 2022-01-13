import os

import pytest

from src.data.FlowerDataset import FlowerDataset

PATH = "data/processed/flowers"
SIZE = "224x224"


@pytest.mark.skipif(not os.path.exists(PATH), reason="Data doesn't exist")
@pytest.mark.parametrize("split", ["train", "test"])
def test_shape(split: str):
    ds = FlowerDataset(PATH, SIZE, split)
    assert len(ds) == (12753 if split == "train" else 7382)

    h, w = SIZE.split("x")
    for img, lbl in ds:
        h = int(h)
        w = int(w)
        assert img.shape == (3, h, w), "Shape mismatch"
        if split == "test":
            assert lbl == None, "Label is not None"
        else:
            assert lbl.nelement() == 1, "More than 1 label"
