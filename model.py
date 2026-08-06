import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models

class CameraFeatureExtractor(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )
        
        self.compress = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, images):
        features = self.backbone(images)
        return self.compress(features)


class PointPillarEncoder(nn.Module):
    def __init__(self, max_points_per_pillar=32, max_pillars=10000, 
                 out_channels=64, bev_h=160, bev_w=160,
                 point_cloud_range=[0.0, -40.0, 0.3, 70.0, 40.0, 3.0],
                 in_point_features=71): 
        
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.out_channels = out_channels
        self.pc_range = point_cloud_range
        self.x_min, self.y_min, self.z_min, self.x_max, self.y_max, self.z_max = self.pc_range
        
        self.voxel_x = (self.x_max - self.x_min) / bev_h 
        self.voxel_y = (self.y_max - self.y_min) / bev_w 
        
        self.pfn = nn.Sequential(
            nn.Linear(in_point_features, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, points, batch_indices):
        batch_size = int(batch_indices.max().item() + 1)
        
        keep = (points[:, 0] >= self.x_min) & (points[:, 0] <= self.x_max) & \
               (points[:, 1] >= self.y_min) & (points[:, 1] <= self.y_max) & \
               (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)
        
        points = points[keep]
        batch_indices = batch_indices[keep]
        
        grid_col = torch.floor((points[:, 1] - self.y_min) / self.voxel_y).long()
        grid_row = torch.floor((points[:, 0] - self.x_min) / self.voxel_x).long()
        grid_row = (self.bev_h - 1) - grid_row
        
        grid_col = torch.clamp(grid_col, 0, self.bev_w - 1)
        grid_row = torch.clamp(grid_row, 0, self.bev_h - 1)
        
        pillar_indices = batch_indices * (self.bev_h * self.bev_w) + grid_row * self.bev_w + grid_col
        
        pillar_center_y = (grid_col.float() * self.voxel_y) + self.y_min + (self.voxel_y / 2.0)
        pillar_center_x = ((self.bev_h - 1 - grid_row.float()) * self.voxel_x) + self.x_min + (self.voxel_x / 2.0)
        pillar_center_z = (self.z_max + self.z_min) / 2.0
        
        rel_x = points[:, 0] - pillar_center_x
        rel_y = points[:, 1] - pillar_center_y
        rel_z = points[:, 2] - pillar_center_z
        
        point_features = torch.cat([points, rel_x.unsqueeze(1), rel_y.unsqueeze(1), rel_z.unsqueeze(1)], dim=1)
        point_features = self.pfn(point_features)
        
        flat_grid_size = batch_size * self.bev_h * self.bev_w
        bev_flat = torch.zeros((flat_grid_size, self.out_channels), device=points.device, dtype=point_features.dtype)
        scatter_indices = pillar_indices.unsqueeze(1).expand(-1, self.out_channels)
        
        bev_flat.scatter_reduce_(dim=0, index=scatter_indices, src=point_features, reduce='amax', include_self=False)
        
        bev_grid = bev_flat.view(batch_size, self.bev_h, self.bev_w, self.out_channels)
        bev_grid = bev_grid.permute(0, 3, 1, 2).contiguous() 
        
        return bev_grid


class BiFPNBEVDecoder(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        
        self.up_p3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.td_conv3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        
        self.up_p2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.td_conv2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.up_p1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.td_conv1 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.down_p2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bu_conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.down_p3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bu_conv3 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        
        self.agg_up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.agg_up3 = nn.ConvTranspose2d(128, 128, 4, stride=4)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(64 + 64 + 128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.occupancy_head = nn.Conv2d(64, 1, kernel_size=1)
        self.dim_head = nn.Conv2d(64, 3, kernel_size=1)
        self.ori_head = nn.Conv2d(64, 2, kernel_size=1)
        
        # --- NEW: CenterPoint Sub-Pixel Offset Head ---
        self.offset_head = nn.Conv2d(64, 2, kernel_size=1)
        
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.occupancy_head.bias, bias_value)

    def forward(self, bev_features):
        p1 = self.enc1(bev_features)         
        p2 = self.enc2(p1)                   
        p3 = self.enc3(p2)                   
        
        b = self.bottleneck(p3)              
        
        td3 = self.td_conv3(p3 + self.up_p3(b))       
        td2 = self.td_conv2(p2 + self.up_p2(td3))     
        td1 = self.td_conv1(p1 + self.up_p1(td2))     
        
        bu1 = td1                                     
        bu2 = self.bu_conv2(td2 + self.down_p2(bu1))  
        bu3 = self.bu_conv3(td3 + self.down_p3(bu2))  
        
        agg1 = bu1                                    
        agg2 = self.agg_up2(bu2)                      
        agg3 = self.agg_up3(bu3)                      
        
        fused = torch.cat([agg1, agg2, agg3], dim=1)  
        out = self.final_conv(fused)                  
        
        return {
            'bev_occupancy': self.occupancy_head(out),
            'dimensions': self.dim_head(out),
            'orientation': self.ori_head(out),
            'offset': self.offset_head(out)
        }


class WaymoBEVDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.camera_encoder = CameraFeatureExtractor(out_channels=64)
        
        self.point_pillar_encoder = PointPillarEncoder(
            out_channels=64, bev_h=160, bev_w=160,
            point_cloud_range=[0.0, -40.0, 0.3, 70.0, 40.0, 3.0],
            in_point_features=71
        )
        
        self.bev_head = BiFPNBEVDecoder(in_channels=64)
            
    def forward(self, lidar_points, batch_indices, camera_images, lidar_uvs):
        batch_size = camera_images.shape[0]
        
        image_features = self.camera_encoder(camera_images)
        
        u_norm = (lidar_uvs[:, 0] / 960.0) * 2.0 - 1.0
        v_norm = (lidar_uvs[:, 1] / 640.0) * 2.0 - 1.0
        
        painted_points_list = []
        
        for b in range(batch_size):
            b_mask = (batch_indices == b)
            b_points = lidar_points[b_mask]
            
            b_grid = torch.stack([u_norm[b_mask], v_norm[b_mask]], dim=-1).view(1, 1, -1, 2)
            
            b_feats = image_features[b:b+1] 
            sampled = F.grid_sample(b_feats, b_grid, align_corners=True, padding_mode='zeros') 
            sampled = sampled.squeeze(0).squeeze(1).transpose(0, 1) 
            
            b_fused = torch.cat([b_points, sampled], dim=1) 
            painted_points_list.append(b_fused)
            
        fused_lidar_points = torch.cat(painted_points_list, dim=0)
        
        bev_features = self.point_pillar_encoder(fused_lidar_points, batch_indices) 
        predictions = self.bev_head(bev_features)
        
        return predictions