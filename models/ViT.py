import torch.nn as nn
import kornia.contrib as K

class ViT(nn.Module):
    def __init__(self, img_size, in_channels, targets):
        super().__init__()
        
        # We define the model
        self.layers = nn.Sequential(
            K.VisionTransformer(image_size=img_size, in_channels=in_channels),
            K.ClassificationHead(num_classes=targets)
        )
    
    def forward(self, x):
        return self.layers(x)