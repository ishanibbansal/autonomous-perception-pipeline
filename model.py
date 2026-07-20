import torch
import torch.nn as nn
from ultralytics import YOLO

class Head3D(nn.Module):
    """
    Custom 3D Detection Head taking backbone features (256 channels)
    and projecting them into 3D bounding box coordinates + classification.
    """
    def __init__(self, in_channels=256, num_classes=3):
        super().__init__()
        
        # Shared feature refinement layer
        self.conv_shared = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )
        
        # Branch 1: Object Classification (num_classes logits)
        self.cls_head = nn.Conv2d(128, num_classes, kernel_size=1)
        
        # Branch 2: 3D Center Location (X, Y, Z depth offset in meters)
        self.loc_head = nn.Conv2d(128, 3, kernel_size=1)
        
        # Branch 3: 3D Box Dimensions (Length, Width, Height)
        self.dim_head = nn.Conv2d(128, 3, kernel_size=1)
        
        # Branch 4: Heading Angle (sin(yaw), cos(yaw)) for stable angle regression
        self.orient_head = nn.Conv2d(128, 2, kernel_size=1)

    def forward(self, x):
        feat = self.conv_shared(x)
        
        cls_preds = self.cls_head(feat)       # Shape: [B, num_classes, 40, 60]
        loc_preds = self.loc_head(feat)       # Shape: [B, 3, 40, 60]
        dim_preds = self.dim_head(feat)       # Shape: [B, 3, 40, 60]
        orient_preds = self.orient_head(feat) # Shape: [B, 2, 40, 60]
        
        return {
            'class': cls_preds,
            'location': loc_preds,
            'dimensions': dim_preds,
            'orientation': orient_preds
        }

class Waymo3DDetector(nn.Module):
    def __init__(self, yolo_version='yolov8n.pt', num_classes=3):
        super().__init__()
        print(f"Loading pre-trained {yolo_version} backbone...")
        
        # 1. Load YOLO model & isolate backbone (Layers 0-9)
        base_yolo = YOLO(yolo_version)
        self.backbone = nn.Sequential(*list(base_yolo.model.model.children())[:10])
        
        # 2. Freeze backbone weights for initial training transfer
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3. Attach Custom 3D Detection Head
        self.head3d = Head3D(in_channels=256, num_classes=num_classes)
            
    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head3d(features)
        return predictions

if __name__ == '__main__':
    model = Waymo3DDetector()
    dummy_input = torch.randn(1, 3, 1280, 1920)
    
    print("\nExecuting forward pass with custom 3D heads...")
    outputs = model(dummy_input)
    
    print("\n--- Output Tensor Shapes ---")
    for key, tensor in outputs.items():
        print(f"{key.capitalize():<12} Head Shape: {tensor.shape}")