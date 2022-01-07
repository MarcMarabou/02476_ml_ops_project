import torch.nn as nn
import kornia.contrib as K

class ViT(nn.Module):
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
    def __init__(self):
        super().__init__()
        
        # We define the model
        self.layers = nn.Sequential(
            K.VisionTransformer(),
            K.ClassificationHead(num_classes=104)
        )
    
    def forward(self, x):
        return self.layers(x)