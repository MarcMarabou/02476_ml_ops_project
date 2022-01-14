import kornia.contrib as K
import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn, optim
import torch.nn.functional as F


class ViT(LightningModule):
    """Vision Transformer Model

    Paramters:
        VisionTransformer:
            image_size (int) - the size of the input image. Default: 224.
            patch_size (int) - the size of the patch to compute the embedding. Default: 16.
            in_channels (int) the number of channels for the input. Default: 3.
            embed_dim (int) - the embedding dimension inside the transformer encoder. Default: 768.
            depth (int) - the depth of the transformer. Default 12.
            num_heads (int) - the number of attention heads. Default 12.
            dropout_rate (float) - dropout rate. Default 0.0.
            dropout_attn (float) - attention dropout rate. Default 0.0.
            backbone (Optional[Module]) - an nn.Module to compute the image patches embeddings. Default: None

        Classification Head:
            embed_dim (int) - the embedding dimension inside the transformer encoder. Default: 768.
            num_classes (int) - an integer representing the number of classes to classify. Default: 10."""

    def __init__(self, args):
        super().__init__()

        self.args = args

        # We define the model
        self.ViT = nn.Sequential(
            K.VisionTransformer(
                image_size=args.image_size,
                patch_size=args.patch_size,
                embed_dim=args.embed_dim,
                num_heads=args.num_heads,
                dropout_rate=args.dropout_rate,
                dropout_attn=args.dropout_attn,
                depth=args.depth,
            ),
            K.ClassificationHead(
                embed_size=args.embed_dim, num_classes=args.num_classes
            ),
        )

        # We define the criterium
        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.ViT(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images
        output = self(images)
        loss = self.criterium(output, labels)
        preds = F.log_softmax(output, 1)
        acc = (labels == preds.argmax(dim=1)).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images
        output = self(images)
        loss = self.criterium(output, labels)
        preds = F.log_softmax(output, 1)
        acc = (labels == preds.argmax(dim=1)).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def predict_step(self, batch, batch_idx):
        images, _ = batch
        preds = self(images)
        return preds

    def configure_optimizers(self):
        args = self.args
        return optim.Adam(self.parameters(), lr=args.lr)
