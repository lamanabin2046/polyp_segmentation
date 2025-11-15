import segmentation_models_pytorch as smp
import torch.nn as nn

def build_unetpp_model():
    """
    Builds a UNet++ segmentation model using SMP.
    Encoder: EfficientNet-b3 (strong + lightweight)
    Classes: 1 (polyp), because it's binary segmentation
    """
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None  # we apply sigmoid later
    )
    return model


# Optional: hybrid loss function
import torch
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2 * intersection + self.eps) / (preds.sum() + targets.sum() + self.eps)
        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        bce = F.binary_cross_entropy(preds, targets, reduction='mean')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt)**self.gamma * bce
        return focal


class HybridLoss(nn.Module):
    """
    Loss = 0.6 * Dice + 0.4 * Focal Loss
    """
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()

    def forward(self, preds, targets):
        return 0.6 * self.dice(preds, targets) + 0.4 * self.focal(preds, targets)
