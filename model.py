import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ultralytics import YOLO

class BEVDecoder(nn.Module):
    """
    Transforms perspective-view features into a top-down BEV occupancy grid.
    Upsamples a 20x30 feature map into a 160x160 spatial grid.
    """
    def __init__(self, in_channels=256):
        super().__init__()
        
        # Step 1: Spatial Transformer 
        # Maps the 3:2 aspect ratio of the camera to a 1:1 square for the BEV grid
        self.perspective_to_bev = nn.AdaptiveAvgPool2d((20, 20))
        
        # Step 2: Upsample 20x20 -> 40x40
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Step 3: Upsample 40x40 -> 80x80
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Step 4: Upsample 80x80 -> 160x160
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Step 5: Final mapping to a single-channel occupancy logit
        self.occupancy_head = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
        # ---> PRIOR PROBABILITY BIAS INITIALIZATION <---
        # Set the bias so the network predicts a 0.01 probability of a vehicle at initialization.
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.occupancy_head.bias, bias_value)

    def forward(self, x):
        # x is [B, 256, 20, 30] from YOLO backbone
        x = self.perspective_to_bev(x)  # -> [B, 256, 20, 20]
        x = self.up1(x)                 # -> [B, 128, 40, 40]
        x = self.up2(x)                 # -> [B, 64, 80, 80]
        x = self.up3(x)                 # -> [B, 32, 160, 160]
        
        # We output raw logits (no sigmoid) for numerical stability during training
        logits = self.occupancy_head(x) # -> [B, 1, 160, 160]
        
        return {'bev_occupancy': logits}

class WaymoBEVDetector(nn.Module):
    def __init__(self, yolo_version='yolov8n.pt'):
        super().__init__()
        print(f"Loading pre-trained {yolo_version} backbone...")
        
        base_yolo = YOLO(yolo_version)
        self.backbone_modules = list(base_yolo.model.model.children())[:10]
        self.backbone = nn.Sequential(*self.backbone_modules)
        
        # ---> 4-CHANNEL EARLY FUSION STEM <---
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
                    if idx == 0:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    param.requires_grad = True
            
        # Swap out the 3D head for the new BEV Decoder
        self.bev_head = BEVDecoder(in_channels=256)
            
    def forward(self, x):
        features = self.backbone(x)
        predictions = self.bev_head(features)
        return predictions

if __name__ == '__main__':
    model = WaymoBEVDetector()
    dummy_input = torch.randn(1, 4, 640, 960)
    
    print("\nExecuting forward pass with BEV fusion architecture...")
    outputs = model(dummy_input)
    
    print("\n--- Output Tensor Shapes ---")
    for key, tensor in outputs.items():
        print(f"{key.capitalize():<20} Shape: {tensor.shape}")