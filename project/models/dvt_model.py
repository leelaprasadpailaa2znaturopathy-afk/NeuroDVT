import torch
import torch.nn as nn
from .vit_model import ViT_Backbone
from .dendritic_layer import DendriticClassifier

class DVT(nn.Module):
    """
    Dendritic Learning-Incorporated Vision Transformer (DVT)
    Combines ViT feature extraction with a Dendritic Neural Network classifier.
    """
    def __init__(self, 
                 image_size=32, 
                 patch_size=4, 
                 num_classes=10, 
                 dim=128, 
                 depth=6, 
                 heads=8, 
                 mlp_dim=256, 
                 channels=3, 
                 dim_head=32, 
                 dropout=0.1, 
                 emb_dropout=0.1,
                 num_branches=16):
        super().__init__()
        
        # 1. Vision Transformer Backbone (Feature Extractor)
        self.vit_backbone = ViT_Backbone(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout
        )
        
        # 2. Dendritic Neural Network (Classifier)
        # Instead of a standard MLP: nn.Linear(dim, num_classes)
        self.classifier = DendriticClassifier(
            input_dim=dim,
            num_classes=num_classes,
            num_branches=num_branches
        )
        
    def forward(self, img):
        # Extract features using ViT
        features = self.vit_backbone(img) # (batch, dim)
        
        # Classify using Dendritic Layer
        logits = self.classifier(features) # (batch, num_classes)
        
        return logits

def create_dvt_model(num_classes=10, image_size=32):
    return DVT(
        image_size=image_size,
        patch_size=4,
        num_classes=num_classes,
        dim=128,
        depth=6,
        heads=8,
        mlp_dim=256,
        num_branches=16
    )
