import torch
import torch.nn as nn

class Waymo3DLoss(nn.Module):
    def __init__(self, cls_weight=1.0, loc_weight=5.0, dim_weight=5.0, orient_weight=2.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight
        self.dim_weight = dim_weight
        self.orient_weight = orient_weight
        
        # Loss functions
        self.cls_loss_fn = nn.BCEWithLogitsLoss() 
        self.smooth_l1_none = nn.SmoothL1Loss(reduction='none')        

    def forward(self, predictions, targets):
        """
        predictions: Dictionary of prediction tensors from Waymo3DDetector
        targets: Dictionary of target tensors including 'mask'
        """
        mask = targets['mask']  # Shape: [Batch, 1, 40, 60]
        
        # 1. Classification Loss (Computed over the ENTIRE grid so it learns background vs object)
        cls_loss = self.cls_loss_fn(predictions['class'], targets['class'])
        
        # Expand mask to match regression channel dimensions
        mask_3d = mask.expand(-1, 3, -1, -1)  # For 3-channel loc and dim
        mask_2d = mask.expand(-1, 2, -1, -1)  # For 2-channel orientation
        
        # Avoid division by zero if a batch has zero valid objects
        num_active = mask.sum().clamp(min=1.0)
        
        # 2. Location Loss (Computed ONLY where objects exist)
        loc_elem = self.smooth_l1_none(predictions['location'], targets['location'])
        loc_loss = (loc_elem * mask_3d).sum() / (num_active * 3.0 + 1e-6)
        
        # 3. Dimension Loss (Computed ONLY where objects exist)
        dim_elem = self.smooth_l1_none(predictions['dimensions'], targets['dimensions'])
        dim_loss = (dim_elem * mask_3d).sum() / (num_active * 3.0 + 1e-6)
        
        # 4. Orientation Loss (Computed ONLY where objects exist)
        orient_elem = self.smooth_l1_none(predictions['orientation'], targets['orientation'])
        orient_loss = (orient_elem * mask_2d).sum() / (num_active * 2.0 + 1e-6)
        
        # Sum the weighted losses
        total_loss = (
            self.cls_weight * cls_loss +
            self.loc_weight * loc_loss +
            self.dim_weight * dim_loss +
            self.orient_weight * orient_loss
        )
        
        return total_loss, {
            'cls_loss': cls_loss.item(),
            'loc_loss': loc_loss.item(),
            'dim_loss': dim_loss.item(),
            'orient_loss': orient_loss.item()
        }

if __name__ == '__main__':
    criterion = Waymo3DLoss()
    batch_size = 1
    grid_h, grid_w = 40, 60
    
    dummy_preds = {
        'class': torch.randn(batch_size, 3, grid_h, grid_w),
        'location': torch.randn(batch_size, 3, grid_h, grid_w),
        'dimensions': torch.randn(batch_size, 3, grid_h, grid_w),
        'orientation': torch.randn(batch_size, 2, grid_h, grid_w)
    }
    
    dummy_targets = {
        'class': torch.zeros(batch_size, 3, grid_h, grid_w),
        'location': torch.randn(batch_size, 3, grid_h, grid_w),
        'dimensions': torch.randn(batch_size, 3, grid_h, grid_w),
        'orientation': torch.randn(batch_size, 2, grid_h, grid_w),
        'mask': torch.ones(batch_size, 1, grid_h, grid_w)
    }
    
    total_loss, loss_dict = criterion(dummy_preds, dummy_targets)
    print(f"\nMasked Loss Test Passed! Total Loss: {total_loss.item():.4f}")