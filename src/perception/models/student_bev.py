import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import math

from src.perception.models.teacher_bev import BiFPNBEVDecoder 

# --- 1. Squeeze-and-Excitation Block ---
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, max(1, in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, in_channels // reduction), in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class LiftSplatViewTransformer(nn.Module):
    def __init__(self, in_channels=256, out_channels=64, bev_h=160, bev_w=160, 
                 d_min=2.0, d_max=50.0, d_bins=48, 
                 x_range=(0.0, 70.0), y_range=(-40.0, 40.0)):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.d_min = d_min
        self.d_max = d_max
        self.d_bins = d_bins
        
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.x_step = (self.x_max - self.x_min) / self.bev_h
        self.y_step = (self.y_max - self.y_min) / self.bev_w
        
        self.depth_net = nn.Conv2d(in_channels, d_bins + out_channels, kernel_size=1)
        self.register_buffer('frustum', self._create_frustum(80, 120, 1280, 1920))
        
    def _create_frustum(self, feat_h, feat_w, image_h, image_w):
        w = torch.linspace(0, image_w - 1, feat_w)
        h = torch.linspace(0, image_h - 1, feat_h)
        d = torch.linspace(self.d_min, self.d_max, self.d_bins)
        
        d_grid, h_grid, w_grid = torch.meshgrid(d, h, w, indexing='ij')
        frustum = torch.stack((w_grid, h_grid, d_grid), dim=-1)
        return frustum
        
    def get_geometry(self, intrinsics, extrinsics_inv):
        B = intrinsics.shape[0]
        
        with torch.autocast('cuda', enabled=False):
            intrinsics = intrinsics.float()
            extrinsics_inv = extrinsics_inv.float()
            points = self.frustum.float().unsqueeze(0).expand(B, -1, -1, -1, -1) 
            
            u = points[..., 0]
            v = points[..., 1]
            d = points[..., 2]
            
            f_u = torch.clamp(intrinsics[:, 0, 0].view(B, 1, 1, 1), min=1e-5)
            f_v = torch.clamp(intrinsics[:, 1, 1].view(B, 1, 1, 1), min=1e-5)
            c_u = intrinsics[:, 0, 2].view(B, 1, 1, 1)
            c_v = intrinsics[:, 1, 2].view(B, 1, 1, 1)
            
            X_cv = (u - c_u) * d / f_u
            Y_cv = (v - c_v) * d / f_v
            Z_cv = d
            
            X_w = Z_cv
            Y_w = -X_cv
            Z_w = -Y_cv
            
            cam_coords_waymo = torch.stack([X_w, Y_w, Z_w], dim=-1)
            cam_coords_hom = torch.cat((cam_coords_waymo, torch.ones_like(cam_coords_waymo[..., :1])), dim=-1)
            
            extrinsics_inv_reshaped = extrinsics_inv.view(B, 1, 1, 1, 4, 4)
            veh_coords_hom = (extrinsics_inv_reshaped @ cam_coords_hom.unsqueeze(-1)).squeeze(-1)
            
        return veh_coords_hom[..., :3] 
        
    def voxel_pooling(self, geom, volume):
        B, D, H, W, C = volume.shape
        
        geom = geom.reshape(B, -1, 3)
        volume = volume.float().reshape(B, -1, C) 
        
        x_idx = ((geom[..., 0] - self.x_min) / self.x_step).long()
        y_idx = ((geom[..., 1] - self.y_min) / self.y_step).long()
        
        x_idx = (self.bev_h - 1) - x_idx
        y_idx = (self.bev_w - 1) - y_idx
        
        valid_mask = (x_idx >= 0) & (x_idx < self.bev_h) & \
                     (y_idx >= 0) & (y_idx < self.bev_w) & \
                     (geom[..., 2] > -5.0) & (geom[..., 2] < 5.0)
        
        bev_maps = []
        for b in range(B):
            v_feat = volume[b][valid_mask[b]]
            x_b = x_idx[b][valid_mask[b]]
            y_b = y_idx[b][valid_mask[b]]
            
            indices = x_b * self.bev_w + y_b 
            
            out = torch.zeros(self.bev_h * self.bev_w, C, device=volume.device, dtype=volume.dtype)
            out.scatter_add_(0, indices.unsqueeze(-1).expand(-1, C), v_feat)
            
            bev_maps.append(out.reshape(self.bev_h, self.bev_w, C).permute(2, 0, 1)) 
            
        return torch.stack(bev_maps, dim=0)
        
    def forward(self, x, intrinsics, extrinsics_inv):
        x = self.depth_net(x)
        depth_logits = x[:, :self.d_bins]
        depth = depth_logits.softmax(dim=1) 
        context = x[:, self.d_bins:]              
        
        volume = depth.unsqueeze(2) * context.unsqueeze(1) 
        volume = volume.permute(0, 1, 3, 4, 2)             
        
        geom = self.get_geometry(intrinsics, extrinsics_inv)
        bev_features = self.voxel_pooling(geom, volume)
        
        return bev_features, depth_logits

class TemporalBEVFusion(nn.Module):
    """
    Memory-Safe Temporal Fusion Module (History Caching).
    Aligns past BEV features to the current ego-vehicle coordinate system
    and fuses them using a detached computational graph to prevent VRAM overflow.
    """
    def __init__(self, in_channels=64):
        super().__init__()
        # Reduces the concatenated [Current + Past] channels back to the original size
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Grid boundaries to map physical meters to [-1, 1] for grid_sample
        self.x_extent = 70.0  # 0 to 70m
        self.y_extent = 80.0  # -40 to 40m (total 80m span)

    def align_past_features(self, past_features, current_extrinsic, past_extrinsic):
        B, C, H, W = past_features.shape
        
        # 1. Compute relative transformation: T_rel = inv(T_current) @ T_past
        # This tells us how the world moved relative to the ego-vehicle.
        rel_transform = torch.inverse(current_extrinsic) @ past_extrinsic
        
        affine_matrices = torch.zeros(B, 2, 3, device=past_features.device)
        
        for b in range(B):
            T = rel_transform[b]
            
            # Extract 2D translation and rotation for the BEV plane (X, Y)
            # Waymo: X is forward, Y is left.
            theta = torch.atan2(T[1, 0], T[0, 0])
            
            # Normalize translations by the physical grid extents to map to [-1, 1] grid space
            tx = T[0, 3] / (self.x_extent / 2.0)
            ty = T[1, 3] / (self.y_extent / 2.0)
            
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            
            # Build the 2x3 affine matrix for PyTorch grid_sample
            affine_matrices[b, 0, 0] = cos_t
            affine_matrices[b, 0, 1] = -sin_t
            affine_matrices[b, 0, 2] = tx
            affine_matrices[b, 1, 0] = sin_t
            affine_matrices[b, 1, 1] = cos_t
            affine_matrices[b, 1, 2] = ty

        # 2. Generate warp grid and resample past features
        # align_corners=False is mathematically preferred for bounding box tasks
        grid = F.affine_grid(affine_matrices, past_features.size(), align_corners=False)
        aligned_past = F.grid_sample(past_features, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        return aligned_past

    def forward(self, current_features, past_features, current_extrinsic, past_extrinsic):
        if past_features is None:
            # If there is no history (e.g., the very first frame of a sequence), 
            # pad with a blank zero-tensor of the exact same shape.
            aligned_past = torch.zeros_like(current_features)
        else:
            # [CRITICAL MEMORY SHIELD]: Detach the past features! 
            # This prevents PyTorch from backpropagating through the previous timestep's ResNet.
            past_features = past_features.detach()
            
            # Physically rotate and shift the past feature map to match current ego-position
            aligned_past = self.align_past_features(past_features, current_extrinsic, past_extrinsic)

        # Concatenate along the channel dimension (e.g., 64 + 64 = 128)
        fused = torch.cat([current_features, aligned_past], dim=1)
        
        # Compress back down to standard feature depth (64)
        return self.fusion_conv(fused)

class StudentBEVDetector(nn.Module):
    def __init__(self, bev_h=160, bev_w=160, feature_channels=64, use_temporal=False):
        super().__init__()
        self.use_temporal = use_temporal
        
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone_stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        ) 
        self.reduce_channel = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.view_transformer = LiftSplatViewTransformer(
            in_channels=256, out_channels=feature_channels, bev_h=bev_h, bev_w=bev_w
        )
        
        # 1. Feature Adaptation Layer
        self.adaptation = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )

        # 2. Squeeze-and-Excitation Channel Attention
        self.se_block = SEBlock(in_channels=feature_channels)
        
        # 3. Conditional Temporal Fusion (Only active for Phase 2)
        if self.use_temporal:
            self.temporal_fusion = TemporalBEVFusion(in_channels=feature_channels)
            
        self.bev_head = BiFPNBEVDecoder(in_channels=feature_channels)

    def forward(self, camera_images, intrinsics, extrinsics_inv, past_features=None, current_extrinsics=None, past_extrinsics=None):
        x = self.backbone_stem(camera_images)
        x = self.reduce_channel(x)
        
        raw_bev_features, depth_logits = self.view_transformer(x, intrinsics, extrinsics_inv)
        
        adapted_features = self.adaptation(raw_bev_features)
        cleaned_features = self.se_block(adapted_features)
        
        # --- Route features based on active Phase ---
        if self.use_temporal:
            fused_features = self.temporal_fusion(
                current_features=cleaned_features,
                past_features=past_features,
                current_extrinsic=current_extrinsics,
                past_extrinsic=past_extrinsics
            )
        else:
            fused_features = cleaned_features
            
        head_outputs = self.bev_head(fused_features)
        
        out = {
            'bev_features': fused_features, 
            'depth_logits': depth_logits
        }
        out.update(head_outputs)
        
        return out