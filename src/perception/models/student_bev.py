import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

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
        
        # [FIX]: Initialize frustum for 80x120 feature map (1/16th of 1280x1920)
        self.register_buffer('frustum', self._create_frustum(80, 120, 1280, 1920))
        
    def _create_frustum(self, feat_h, feat_w, image_h, image_w):
        w = torch.linspace(0, image_w - 1, feat_w)
        h = torch.linspace(0, image_h - 1, feat_h)
        d = torch.linspace(self.d_min, self.d_max, self.d_bins)
        
        d_grid, h_grid, w_grid = torch.meshgrid(d, h, w, indexing='ij')
        frustum = torch.stack((w_grid, h_grid, d_grid), dim=-1)
        return frustum
        
    def get_geometry(self, intrinsics, extrinsics):
        B = intrinsics.shape[0]
        points = self.frustum.unsqueeze(0).expand(B, -1, -1, -1, -1) 
        
        d = points[..., 2:3]
        uv = points[..., 0:2] * d
        uvd = torch.cat((uv, d), dim=-1)
        
        # [FIX]: Safely align 3x3 and 4x4 matrices with the 5D frustum tensor
        intrinsics_inv = torch.inverse(intrinsics).view(B, 1, 1, 1, 3, 3)
        extrinsics_v = extrinsics.view(B, 1, 1, 1, 4, 4)
        
        cam_coords = (intrinsics_inv @ uvd.unsqueeze(-1)).squeeze(-1)
        cam_coords_hom = torch.cat((cam_coords, torch.ones_like(cam_coords[..., :1])), dim=-1)
        veh_coords_hom = (extrinsics_v @ cam_coords_hom.unsqueeze(-1)).squeeze(-1)
        
        return veh_coords_hom[..., :3] 
        
    def voxel_pooling(self, geom, volume):
        B, D, H, W, C = volume.shape
        
        # [FIX]: Use reshape instead of view to prevent permuted memory crashes
        geom = geom.reshape(B, -1, 3)
        volume = volume.reshape(B, -1, C)
        
        x_idx = ((geom[..., 0] - self.x_min) / self.x_step).long()
        y_idx = ((geom[..., 1] - self.y_min) / self.y_step).long()
        
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
        
    def forward(self, x, intrinsics, extrinsics):
        x = self.depth_net(x)
        depth_logits = x[:, :self.d_bins]
        depth = depth_logits.softmax(dim=1) 
        context = x[:, self.d_bins:]              
        
        volume = depth.unsqueeze(2) * context.unsqueeze(1) 
        volume = volume.permute(0, 1, 3, 4, 2)             
        
        geom = self.get_geometry(intrinsics, extrinsics)
        bev_features = self.voxel_pooling(geom, volume)
        
        return bev_features, depth_logits


class StudentBEVDetector(nn.Module):
    def __init__(self, bev_h=160, bev_w=160, feature_channels=64):
        super().__init__()
        
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone_stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        ) 
        
        self.reduce_channel = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.view_transformer = LiftSplatViewTransformer(
            in_channels=256, 
            out_channels=feature_channels,
            bev_h=bev_h,
            bev_w=bev_w
        )
        
        self.bev_head = nn.Sequential(
            nn.Conv2d(feature_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, camera_images, intrinsics, extrinsics):
        x = self.backbone_stem(camera_images)
        x = self.reduce_channel(x)
        
        bev_features, depth_logits = self.view_transformer(x, intrinsics, extrinsics)
        bev_occupancy = self.bev_head(bev_features)
        
        return {
            'bev_features': bev_features,
            'bev_occupancy': bev_occupancy,
            'depth_logits': depth_logits
        }