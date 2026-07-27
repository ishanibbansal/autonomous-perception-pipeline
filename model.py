import torch
import torch.nn as nn
from ultralytics import YOLO

class Head3D(nn.Module):
    """
    Custom 3D Detection Head with Bin-Based Depth and CenterNet Architecture.
    """
    def __init__(self, in_channels=256, num_classes=3, num_depth_bins=40):
        super().__init__()
        
        self.conv_shared = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )
        
        # Branch 1: Heatmap Object Classification
        self.cls_head = nn.Conv2d(128, num_classes, kernel_size=1)
        
        # Branch 2: 2D Center Location (X offset, Y offset)
        self.offset_head = nn.Conv2d(128, 2, kernel_size=1)
        
        # Branch 3: Bin-Based Depth (Classification + Residual)
        self.depth_bin_head = nn.Conv2d(128, num_depth_bins, kernel_size=1)
        self.depth_res_head = nn.Conv2d(128, 1, kernel_size=1)
        
        # Branch 4: 3D Box Dimensions
        self.dim_head = nn.Conv2d(128, 3, kernel_size=1)
        
        # Branch 5: Heading Angle (sin(yaw), cos(yaw))
        self.orient_head = nn.Conv2d(128, 2, kernel_size=1)

    def forward(self, x):
        feat = self.conv_shared(x)
        
        cls_preds = self.cls_head(feat)                   # [B, C, H, W]
        loc_preds = torch.sigmoid(self.offset_head(feat)) # [B, 2, H, W] constrained to [0, 1)
        
        depth_bin_preds = self.depth_bin_head(feat)       # Logits for CrossEntropy [B, bins, H, W]
        depth_res_preds = torch.sigmoid(self.depth_res_head(feat)) # [B, 1, H, W] constrained to [0, 1)
        
        dim_preds = self.dim_head(feat)                   # [B, 3, H, W]
        orient_preds = self.orient_head(feat)             # [B, 2, H, W]
        
        return {
            'class': cls_preds,
            'location': loc_preds,
            'depth_bin': depth_bin_preds,
            'depth_res': depth_res_preds,
            'dimensions': dim_preds,
            'orientation': orient_preds
        }

class Waymo3DDetector(nn.Module):
    def __init__(self, yolo_version='yolov8n.pt', num_classes=3, num_depth_bins=40):
        super().__init__()
        print(f"Loading pre-trained {yolo_version} backbone...")
        
        base_yolo = YOLO(yolo_version)
        self.backbone_modules = list(base_yolo.model.model.children())[:10]
        self.backbone = nn.Sequential(*self.backbone_modules)
        
        # ---> NEW: 4-CHANNEL EARLY FUSION STEM <---
        old_stem = self.backbone[0].conv
        
        new_stem = nn.Conv2d(
            in_channels=4,  # RGB (3) + LiDAR Depth (1)
            out_channels=old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=(old_stem.bias is not None)
        )
        
        with torch.no_grad():
            # Copy the pretrained RGB weights perfectly into the first 3 channels
            new_stem.weight[:, :3, :, :] = old_stem.weight
            
            # Zero-initialize the 4th (Depth) channel
            new_stem.weight[:, 3:4, :, :] = 0.0
            
            if old_stem.bias is not None:
                new_stem.bias = old_stem.bias
                
        # Swap the newly minted 4-channel stem back into the backbone
        self.backbone[0].conv = new_stem
        # ------------------------------------------
        
        # Freeze initial layers, leave deeper layers and new stem to train
        for idx, module in enumerate(self.backbone):
            for param in module.parameters():
                if idx <= 4:
                    # Note: We must ensure the new stem (idx == 0) is trainable 
                    # so it learns the depth channel representation.
                    if idx == 0:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    param.requires_grad = True
            
        self.head3d = Head3D(in_channels=256, num_classes=num_classes, num_depth_bins=num_depth_bins)
            
    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head3d(features)
        return predictions

if __name__ == '__main__':
    model = Waymo3DDetector()
    # Changed input to 4 channels to test the new Early Fusion architecture
    dummy_input = torch.randn(1, 4, 640, 960)
    
    print("\nExecuting forward pass with custom 4-channel fusion backbone...")
    outputs = model(dummy_input)
    
    print("\n--- Output Tensor Shapes ---")
    for key, tensor in outputs.items():
        print(f"{key.capitalize():<12} Head Shape: {tensor.shape}")