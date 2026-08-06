import torch
import torch.nn as nn
import torch.nn.functional as F

class BEVFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions, targets):
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        probs = torch.sigmoid(predictions)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_weight * torch.pow((1 - p_t), self.gamma)
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedBEVLoss(nn.Module):
    def __init__(self, focal_loss, iou_weight=2.0, smooth=1.0):
        super().__init__()
        self.focal_loss = focal_loss
        self.dice_weight = iou_weight
        self.smooth = smooth 

    def forward(self, predictions, targets):
        focal = self.focal_loss(predictions, targets)
        
        preds_prob = torch.sigmoid(predictions)
        preds_flat = preds_prob.view(preds_prob.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = (preds_flat * targets_flat).sum(-1)
        cardinality = preds_flat.sum(-1) + targets_flat.sum(-1)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = (1.0 - dice_score).mean()
        
        return focal + (self.dice_weight * dice_loss)


class WaymoDetectionLoss(nn.Module):
    """
    Master Loss Wrapper: Accepts full prediction and target dictionaries.
    """
    def __init__(self):
        super().__init__()
        self.occupancy_loss = CombinedBEVLoss(focal_loss=BEVFocalLoss(alpha=0.25, gamma=2.0))
        self.regression_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, predictions, targets):
        # 1. Spatial Occupancy Loss
        occ_loss = self.occupancy_loss(predictions['bev_occupancy'], targets['bev_occupancy'])

        mask = targets['mask'] # (B, 1, 160, 160)
        mask_sum = mask.sum() + 1e-6
        
        # 2. Dimensions (L, W, H)
        dim_err = self.regression_loss(predictions['dimensions'], targets['dimensions'])
        dim_loss = (dim_err * mask).sum() / mask_sum

        # 3. Orientation (Sin, Cos)
        ori_err = self.regression_loss(predictions['orientation'], targets['orientation'])
        ori_loss = (ori_err * mask).sum() / mask_sum
        
        # 4. NEW: Sub-Pixel Center Offset (dx, dy)
        offset_err = self.regression_loss(predictions['offset'], targets['offset'])
        offset_loss = (offset_err * mask).sum() / mask_sum

        # Final Weighted Fusion
        total_loss = occ_loss + (2.0 * dim_loss) + (1.0 * ori_loss) + (1.0 * offset_loss)
        
        return total_loss