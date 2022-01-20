import base64
import json
import subprocess
from io import BytesIO

import requests
import torch
from torchvision.transforms.functional import to_pil_image

from src.data.FlowerDataset import FlowerDataset

if __name__ == "__main__":
    data = FlowerDataset(
        flowers_path="data/processed/flowers", size="224x224", split="val"
    )
    trainloader = torch.utils.data.DataLoader(data, batch_size=20, shuffle=True)

    images, labels = next(iter(trainloader))

    endpoint = "https://europe-west1-ml.googleapis.com/v1/projects/aerobic-datum-337911/models/vit/versions/v1:predict"
    headers = {
        "Authorization": "Bearer {}".format(
            subprocess.check_output(["gcloud", "auth", "print-access-token"])
            .decode("utf-8")
            .replace("\n", "")
        ),
        "Content-Type": "application/json; charset=utf-8",
    }
    payload = {"instances": [{"data": {"b64": None}}]}

    mapping = json.load(open("index_to_name.json", "r"))
    for i in range(len(images)):
        image = images[i]
        label = labels[i]

        buffered = BytesIO()
        to_pil_image(image).save(buffered, format="JPEG")

        payload["instances"][0]["data"]["b64"] = base64.b64encode(
            buffered.getvalue()
        ).decode("utf-8")

        json_payload = json.dumps(payload)
        r = requests.post(endpoint, headers=headers, data=json_payload)

        print("Label: ", mapping[str(label.item())])
        print("Prediction: ", r.text)
