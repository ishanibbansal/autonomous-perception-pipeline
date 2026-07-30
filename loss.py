import torch
import torch.nn as nn
import torch.nn.functional as F

class BEVFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for highly imbalanced Bird's Eye View (BEV) occupancy grids.
        
        Args:
            alpha: Weighting factor for the positive class (vehicles). 
                   Since vehicles are rare, we keep it balanced or slightly favor positives.
            gamma: The focusing parameter. Higher values down-weight easy examples more aggressively.
                   gamma=2.0 is the standard paper recommendation.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, 1, H, W] tensor of raw logits from the BEVDecoder.
            targets: [B, 1, H, W] tensor of ground truth binary occupancy (1.0 or 0.0).
        """
        # 1. Calculate standard Binary Cross-Entropy (BCE) with logits
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        
        # 2. Get the probabilities by applying sigmoid to the logits
        probs = torch.sigmoid(predictions)
        
        # 3. Calculate p_t (the probability of the true class)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 4. Apply the alpha weighting
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 5. Calculate the final focal weight: alpha * (1 - p_t)^gamma
        focal_weight = alpha_weight * torch.pow((1 - p_t), self.gamma)
        
        # 6. Apply weights to the base BCE loss
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedBEVLoss(nn.Module):
    def __init__(self, focal_loss, iou_weight=1.0):
        """
        Combines BEVFocalLoss with a differentiable Soft-IoU loss.
        Focal Loss stabilizes pixel classification; Soft-IoU sharpens box boundaries.
        """
        super().__init__()
        self.focal_loss = focal_loss
        self.iou_weight = iou_weight

    def forward(self, predictions, targets):
        # 1. Standard focal loss
        focal = self.focal_loss(predictions, targets)
        
        # 2. Differentiable Soft-IoU
        preds_prob = torch.sigmoid(predictions)
        
        # Sum across spatial grid dimensions (B, C, H, W)
        intersection = (preds_prob * targets).sum(dim=(-2, -1))
        union = preds_prob.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1)) - intersection
        
        # Add epsilon to prevent division by zero
        soft_iou = (intersection + 1e-6) / (union + 1e-6)
        iou_loss = (1.0 - soft_iou).mean()
        
        return focal + (self.iou_weight * iou_loss)

if __name__ == '__main__':
    print("Testing CombinedBEVLoss...")
    
    # Simulate a batch of 2 grids, 160x160
    dummy_logits = torch.randn(2, 1, 160, 160)
    
    # Create a dummy target mask (mostly zeros, a few ones)
    dummy_targets = torch.zeros(2, 1, 160, 160)
    dummy_targets[:, :, 75:85, 75:85] = 1.0  # Simulate a 10x10 vehicle in the center
    
    base_focal = BEVFocalLoss(alpha=0.25, gamma=2.0)
    criterion = CombinedBEVLoss(focal_loss=base_focal, iou_weight=1.0)
    loss = criterion(dummy_logits, dummy_targets)
    
    print(f"Loss value: {loss.item():.4f}")
    print("SUCCESS: Loss function is ready for the training loop!")