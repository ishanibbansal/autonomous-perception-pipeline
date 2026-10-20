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
    Occupancy uses penalty-reduced CenterNet focal loss, normalized by positive
    peak count. LiDAR sees the full grid, so the focal mask is all ones.
    """
    def __init__(self, focal_alpha=2.0, focal_beta=4.0):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta
        self.regression_loss = nn.SmoothL1Loss(reduction='none')

    def centernet_focal_loss(self, preds, targets):
        preds = torch.clamp(torch.sigmoid(preds), min=1e-4, max=1 - 1e-4)
        pos_inds = targets.eq(1).float()
        neg_inds = targets.lt(1).float()
        neg_weights = torch.pow(1 - targets, self.focal_beta)
        pos_loss = torch.log(preds) * torch.pow(1 - preds, self.focal_alpha) * pos_inds
        neg_loss = torch.log(1 - preds) * torch.pow(preds, self.focal_alpha) * neg_weights * neg_inds
        num_pos = torch.clamp(pos_inds.sum(), min=1.0)
        return -(pos_loss.sum() + neg_loss.sum()) / num_pos

    def forward(self, predictions, targets):
        occupancy = targets['bev_occupancy']
        loss_focal = self.centernet_focal_loss(predictions['bev_occupancy'], occupancy)

        preds_prob = torch.sigmoid(predictions['bev_occupancy'])
        intersection = (preds_prob * occupancy).sum()
        cardinality = preds_prob.sum() + occupancy.sum()
        loss_dice = 1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)

        mask = targets['mask']
        mask_sum = mask.sum() + 1e-6
        
        dim_err = self.regression_loss(predictions['dimensions'], targets['dimensions'])
        loss_dim = (dim_err * mask).sum() / mask_sum

        ori_err = self.regression_loss(predictions['orientation'], targets['orientation'])
        loss_ori = (ori_err * mask).sum() / mask_sum
        
        offset_err = self.regression_loss(predictions['offset'], targets['offset'])
        loss_offset = (offset_err * mask).sum() / mask_sum

        loss_total = loss_focal + (2.0 * loss_dice) + (2.0 * loss_dim) + loss_ori + loss_offset
        return {
            'loss_total': loss_total,
            'loss_focal': loss_focal,
            'loss_dice': loss_dice,
            'loss_dim': loss_dim,
            'loss_ori': loss_ori,
            'loss_offset': loss_offset,
        }