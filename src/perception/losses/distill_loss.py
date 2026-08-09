import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalDistillationLoss(nn.Module):
    def __init__(self, alpha_feat=2.0, alpha_depth=1.0, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.alpha_feat = alpha_feat
        self.alpha_depth = alpha_depth
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        self.feature_loss = nn.SmoothL1Loss()
        self.depth_loss = nn.CrossEntropyLoss(ignore_index=-1) 

    def focal_loss(self, preds, targets):
        bce = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_term = self.focal_alpha * (1 - pt) ** self.focal_gamma
        return (focal_term * bce).mean()
        
    def dice_loss(self, preds, targets, smooth=1e-5):
        preds_prob = torch.sigmoid(preds)
        intersection = (preds_prob * targets).sum(dim=(2, 3))
        union = preds_prob.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
        return dice.mean()

    def forward(self, student_outputs, teacher_features, ground_truth, depth_labels=None):
        student_features = student_outputs['bev_features']
        student_occupancy = student_outputs['bev_occupancy']
        
        # 1. Spatial Target Loss
        loss_focal = self.focal_loss(student_occupancy, ground_truth)
        loss_dice = self.dice_loss(student_occupancy, ground_truth)
        loss_det = loss_focal + loss_dice
        
        # 2. Teacher Feature Mimicking Loss
        loss_feat = self.feature_loss(student_features, teacher_features.detach())
        
        # 3. Explicit Depth Supervision Loss (Single Frame)
        loss_depth = torch.tensor(0.0, device=student_occupancy.device)
        if 'depth_logits' in student_outputs and depth_labels is not None:
            # depth_logits shape: [B, 48, 40, 60], depth_labels shape: [B, 40, 60]
            loss_depth = self.depth_loss(student_outputs['depth_logits'], depth_labels)

        total_loss = loss_det + (self.alpha_feat * loss_feat) + (self.alpha_depth * loss_depth)

        return total_loss, {
            'loss_total': total_loss.item(),
            'loss_det': loss_det.item(),
            'loss_feat': loss_feat.item(),
            'loss_depth': loss_depth.item()
        }