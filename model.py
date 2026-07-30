import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ultralytics import YOLO

class ViewTransformer(nn.Module):
    """
    Transforms perspective camera features into a top-down BEV grid 
    by collapsing the vertical axis (height) and projecting it to depth.
    """
    def __init__(self, in_channels, out_channels, cam_h=20, cam_w=30, bev_h=160, bev_w=160):
        super().__init__()
        self.cam_h = cam_h
        self.cam_w = cam_w
        self.bev_h = bev_h
        self.bev_w = bev_w
        
        # Flatten the channel and height dimensions, then linearly project to BEV depth
        self.fc_depth = nn.Linear(in_channels * cam_h, out_channels * bev_h)
        
        # Spatial refinement in BEV space
        self.bev_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Reorganize: [B, C, H, W] -> [B, W, C, H] -> [B, W, C * H]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, W, C * H)
        
        # Project image height into BEV depth: [B, W, out_channels * bev_h]
        x = self.fc_depth(x)
        
        # Reorganize: [B, W, out_channels * bev_h] -> [B, out_channels, bev_h, W]
        x = x.view(B, W, -1, self.bev_h)
        x = x.permute(0, 2, 3, 1).contiguous()
        
        # Interpolate the width dimension to match BEV width (160)
        x = F.interpolate(x, size=(self.bev_h, self.bev_w), mode='bilinear', align_corners=False)
        
        return self.bev_refine(x)

class BEVDecoder(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        
        # Step 1: Geometric Transformation from Image to BEV Space
        self.view_transformer = ViewTransformer(
            in_channels=in_channels, 
            out_channels=64, 
            cam_h=20, cam_w=30, 
            bev_h=160, bev_w=160
        )
        
        # Step 2: BEV Feature processing with Spatial Dropout
        self.decoder_blocks = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2), # <--- ADDED SPATIAL DROPOUT
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)  # <--- ADDED SPATIAL DROPOUT BEFORE HEAD
        )
        
        # Step 3: Final mapping to a single-channel occupancy logit
        self.occupancy_head = nn.Conv2d(32, 1, kernel_size=1)
        
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.occupancy_head.bias, bias_value)

    def forward(self, x):
        # x is [B, 256, 20, 30]
        bev_features = self.view_transformer(x)    # -> [B, 64, 160, 160]
        bev_features = self.decoder_blocks(bev_features) # -> [B, 32, 160, 160]
        logits = self.occupancy_head(bev_features) # -> [B, 1, 160, 160]
        
        return {'bev_occupancy': logits}

class WaymoBEVDetector(nn.Module):
    def __init__(self, yolo_version='yolov8n.pt'):
        super().__init__()
        print(f"Loading pre-trained {yolo_version} backbone...")
        base_yolo = YOLO(yolo_version)
        self.backbone_modules = list(base_yolo.model.model.children())[:10]
        self.backbone = nn.Sequential(*self.backbone_modules)
        
        old_stem = self.backbone[0].conv
        new_stem = nn.Conv2d(
            in_channels=4, out_channels=old_stem.out_channels,
            kernel_size=old_stem.kernel_size, stride=old_stem.stride,
            padding=old_stem.padding, bias=(old_stem.bias is not None)
        )
        with torch.no_grad():
            new_stem.weight[:, :3, :, :] = old_stem.weight
            new_stem.weight[:, 3:4, :, :] = 0.0
            if old_stem.bias is not None:
                new_stem.bias = old_stem.bias
                
        self.backbone[0].conv = new_stem
        
        for idx, module in enumerate(self.backbone):
            for param in module.parameters():
                if idx <= 7:
                    param.requires_grad = (idx == 0)
                else:
                    param.requires_grad = True
            
        self.bev_head = BEVDecoder(in_channels=256)
            
    def forward(self, x):
        features = self.backbone(x)
        predictions = self.bev_head(features)
        return predictions

if __name__ == '__main__':
    model = WaymoBEVDetector()
    dummy_input = torch.randn(1, 4, 640, 960)
    outputs = model(dummy_input)
    
    print("\n--- Output Tensor Shapes ---")
    for key, tensor in outputs.items():
        print(f"{key.capitalize():<20} Shape: {tensor.shape}")