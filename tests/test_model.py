import argparse

import pytest
import torch
import torch.nn.functional as F

from src.models.train_model import get_args
from src.models.ViT import ViT


# @pytest.mark.skip(reason="Model is not working currently: 01-13-2022")
def test_predictions():
    args = get_args()
    model = ViT(**vars(args))
    model.train()
    randImage = torch.rand([1, 3, 224, 224])
    output = model(randImage)
    preds = F.softmax(output, 1)
    assert torch.all(1.0 >= preds) and torch.all(
        0.0 <= preds
    ), "The model output either does not predict values between 0 and 1"
    # needs to be rounded as the cpu fails to add stuff correctly on the 8-ish decimal place
    assert 1.0 == round(
        torch.sum(preds).item(), 3
    ), "The model predictions does not sum to 1"
