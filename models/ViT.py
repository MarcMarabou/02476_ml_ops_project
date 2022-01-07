import torch.nn as nn
import kornia.contrib as K

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        
        # We define the model
        self.layers = nn.Sequential(
            K.VisionTransformer(),
            K.ClassificationHead(num_classes=104)
        )
    
    def forward(self, x):
        return self.layers(x)