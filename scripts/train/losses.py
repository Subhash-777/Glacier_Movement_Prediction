"""
Loss functions for glacier lake segmentation
Combines Dice, BCE, and Focal losses for class imbalance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        pred_prob = torch.sigmoid(pred)
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        
        alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        focal_loss = alpha_factor * modulating_factor * bce_loss
        
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """Combined loss: Dice + BCE + Focal"""
    def __init__(self, dice_weight=0.5, bce_weight=0.3, focal_weight=0.2,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        total_loss = (self.dice_weight * dice + 
                     self.bce_weight * bce + 
                     self.focal_weight * focal)
        
        return total_loss, {'dice': dice.item(), 'bce': bce.item(), 'focal': focal.item()}
