import torch
import torch.nn as nn

class Waymo3DLoss(nn.Module):
    def __init__(self, cls_weight=1.0, loc_weight=1.5, dim_weight=1.0, orient_weight=1.0):
        super().__init__()
        # Weights to balance the four different gradients so one doesn't overpower the others
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight
        self.dim_weight = dim_weight
        self.orient_weight = orient_weight
        
        # The core mathematical loss functions
        self.cls_loss_fn = nn.BCEWithLogitsLoss() 
        self.smooth_l1 = nn.SmoothL1Loss()        

    def forward(self, predictions, targets):
        """
        predictions: The 4-tensor dictionary output from Waymo3DDetector
        targets: A matching 4-tensor dictionary representing the ground truth labels
        """
        # 1. Classification Loss 
        cls_loss = self.cls_loss_fn(predictions['class'], targets['class'])
        
        # 2. Location Loss (X, Y, Z depth)
        loc_loss = self.smooth_l1(predictions['location'], targets['location'])
        
        # 3. Dimension Loss (Length, Width, Height)
        dim_loss = self.smooth_l1(predictions['dimensions'], targets['dimensions'])
        
        # 4. Orientation Loss (Heading Angle)
        orient_loss = self.smooth_l1(predictions['orientation'], targets['orientation'])
        
        # Sum the weighted losses into one single scalar for backpropagation
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
    # Initialize the custom multi-task loss
    criterion = Waymo3DLoss()
    
    # 1. Create dummy predictions (mimicking the exact output shapes from model.py)
    batch_size = 1
    grid_h, grid_w = 40, 60
    
    dummy_preds = {
        'class': torch.randn(batch_size, 3, grid_h, grid_w),
        'location': torch.randn(batch_size, 3, grid_h, grid_w),
        'dimensions': torch.randn(batch_size, 3, grid_h, grid_w),
        'orientation': torch.randn(batch_size, 2, grid_h, grid_w)
    }
    
    # 2. Create dummy targets matching those shapes
    dummy_targets = {
        'class': torch.randn(batch_size, 3, grid_h, grid_w),
        'location': torch.randn(batch_size, 3, grid_h, grid_w),
        'dimensions': torch.randn(batch_size, 3, grid_h, grid_w),
        'orientation': torch.randn(batch_size, 2, grid_h, grid_w)
    }
    
    # 3. Calculate the loss
    total_loss, loss_dict = criterion(dummy_preds, dummy_targets)
    
    print(f"\nTotal Backpropagation Loss: {total_loss.item():.4f}")
    print("\n--- Gradient Components ---")
    for k, v in loss_dict.items():
        print(f"{k.capitalize():<12}: {v:.4f}")