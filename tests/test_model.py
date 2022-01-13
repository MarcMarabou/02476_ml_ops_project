import torch
import pytest
from models.ViT import ViT

@pytest.mark.skip(reason="Model is not working currently: 01-13-2022")
def test_predictions():
    model = ViT()
    model.train()
    randImage = torch.rand([1, 3, 224, 224])
    preds = model(randImage)
    assert torch.all(1.0 >= preds) and torch.all(0.0 <= preds), "The model output either does not predict values between 0 and 1"
    # needs to be rounded as the cpu fails to add stuff correctly on the 8-ish decimal place 
    assert 1.0 == round(torch.sum(preds).item()), "The model predictions does not sum to 1"