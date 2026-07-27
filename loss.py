import torch
import torch.nn as nn
import torch.nn.functional as F

def centernet_focal_loss(pred, target, alpha=2.0, beta=4.0):
    """Penalty-reduced focal loss for Gaussian heatmaps."""
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    
    neg_weights = torch.pow(1 - target, beta)
    
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds
    
    num_pos = pos_inds.float().sum()
    if num_pos == 0:
        return -neg_loss.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos

class Waymo3DLoss(nn.Module):
    def __init__(self, cls_weight=1.0, loc_weight=5.0, depth_weight=1.0, dim_weight=5.0, orient_weight=2.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight
        self.depth_weight = depth_weight
        self.dim_weight = dim_weight
        self.orient_weight = orient_weight
        
        self.depth_bin_loss_fn = nn.CrossEntropyLoss(reduction='none') 
        self.smooth_l1_none = nn.SmoothL1Loss(reduction='none')        

    def forward(self, predictions, targets):
        mask = targets['mask']  # Shape: [Batch, 1, H, W]
        num_active = mask.sum().clamp(min=1.0)
        
        # 1. Heatmap Focal Loss (computed over entire grid)
        cls_loss = centernet_focal_loss(predictions['class'], targets['class'])
        
        # Expand masks
        mask_2d = mask.expand(-1, 2, -1, -1)
        mask_3d = mask.expand(-1, 3, -1, -1)
        
        # 2. 2D Center Location Loss
        loc_elem = self.smooth_l1_none(predictions['location'], targets['location'])
        loc_loss = (loc_elem * mask_2d).sum() / (num_active * 2.0 + 1e-6)
        
        # 3. Bin-Based Depth Loss (Classification + Residual)
        # CrossEntropy expects [B, Classes, H, W] and targets [B, H, W] (LongTensor)
        depth_bin_elem = self.depth_bin_loss_fn(predictions['depth_bin'], targets['depth_bin'])
        depth_bin_loss = (depth_bin_elem * mask.squeeze(1)).sum() / num_active
        
        depth_res_elem = self.smooth_l1_none(predictions['depth_res'], targets['depth_res'])
        depth_res_loss = (depth_res_elem * mask).sum() / num_active
        
        total_depth_loss = depth_bin_loss + depth_res_loss
        
        # 4. Dimension & Orientation Loss
        dim_elem = self.smooth_l1_none(predictions['dimensions'], targets['dimensions'])
        dim_loss = (dim_elem * mask_3d).sum() / (num_active * 3.0 + 1e-6)
        
        orient_elem = self.smooth_l1_none(predictions['orientation'], targets['orientation'])
        orient_loss = (orient_elem * mask_2d).sum() / (num_active * 2.0 + 1e-6)
        
        total_loss = (
            self.cls_weight * cls_loss +
            self.loc_weight * loc_loss +
            self.depth_weight * total_depth_loss +
            self.dim_weight * dim_loss +
            self.orient_weight * orient_loss
        )
        
        return total_loss, {
            'cls_loss': cls_loss.item(),
            'loc_loss': loc_loss.item(),
            'depth_loss': total_depth_loss.item(),
            'dim_loss': dim_loss.item(),
            'orient_loss': orient_loss.item()
        }