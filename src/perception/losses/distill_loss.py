import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalDistillationLoss(nn.Module):
    def __init__(self, alpha_feat=0.1, alpha_depth=0.1, focal_alpha=2.0, focal_beta=4.0):
        super().__init__()
        self.alpha_feat = alpha_feat
        self.alpha_depth = alpha_depth
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta
        
        self.feature_loss = nn.SmoothL1Loss(reduction='none')
        self.depth_loss = nn.CrossEntropyLoss(ignore_index=-1) 
        
        self.register_buffer('fov_mask', self._create_fov_mask(160, 160))

    def _create_fov_mask(self, bev_h, bev_w):
        mask = torch.zeros(1, 1, bev_h, bev_w)
        for i in range(bev_h):
            x = 70.0 - (i / float(bev_h)) * 70.0
            x = max(x, 1.0) 
            for j in range(bev_w):
                y = 40.0 - (j / float(bev_w)) * 80.0
                if abs(y / x) < 0.65: 
                    mask[0, 0, i, j] = 1.0
        return mask

    def centernet_focal_loss(self, preds, targets, mask):
        preds = torch.clamp(torch.sigmoid(preds), min=1e-4, max=1 - 1e-4)
        
        pos_inds = targets.eq(1).float() * mask
        neg_inds = targets.lt(1).float() * mask
        neg_weights = torch.pow(1 - targets, self.focal_beta)
        
        pos_loss = torch.log(preds) * torch.pow(1 - preds, self.focal_alpha) * pos_inds
        neg_loss = torch.log(1 - preds) * torch.pow(preds, self.focal_alpha) * neg_weights * neg_inds
        
        num_pos = pos_inds.sum()
        num_pos = torch.clamp(num_pos, min=1.0)
        
        return -(pos_loss.sum() + neg_loss.sum()) / num_pos
        
    def dice_loss(self, preds, targets, mask, smooth=1e-5):
        preds_prob = torch.sigmoid(preds) * mask
        targets_masked = targets * mask
        
        intersection = (preds_prob * targets_masked).sum(dim=(2, 3))
        union = preds_prob.sum(dim=(2, 3)) + targets_masked.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
        return dice.mean()

    def forward(self, student_outputs, teacher_features, ground_truth, depth_labels=None):
        # [CRITICAL FIX]: Force all inputs up to FP32 before calculating the massive custom sums!
        # This completely eliminates the 65,504 overflow limit from the AMP autocast.
        student_features = student_outputs['bev_features'].float()
        student_occupancy = student_outputs['bev_occupancy'].float()
        teacher_features = teacher_features.float()
        ground_truth = ground_truth.float()
        
        b_size = student_occupancy.shape[0]
        fov_mask = self.fov_mask.expand(b_size, -1, -1, -1).float()
        
        loss_focal = self.centernet_focal_loss(student_occupancy, ground_truth, fov_mask)
        loss_dice = self.dice_loss(student_occupancy, ground_truth, fov_mask)
        loss_det = loss_focal + loss_dice
        
        feat_err = self.feature_loss(student_features, teacher_features.detach())
        loss_feat = (feat_err * fov_mask).sum() / (fov_mask.sum() * student_features.shape[1] + 1e-6)
        
        loss_depth = torch.tensor(0.0, device=student_occupancy.device, dtype=torch.float32)
        if 'depth_logits' in student_outputs and depth_labels is not None:
            loss_depth = self.depth_loss(student_outputs['depth_logits'], depth_labels)

        total_loss = loss_det + (self.alpha_feat * loss_feat) + (self.alpha_depth * loss_depth)

        return total_loss, {
            'loss_total': total_loss.item(),
            'loss_det': loss_det.item(),
            'loss_feat': loss_feat.item(),
            'loss_depth': loss_depth.item()
        }