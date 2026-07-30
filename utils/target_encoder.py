import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import math
import cv2
import numpy as np

# [TargetEncoder class remains exactly the same]
class TargetEncoder:
    def __init__(self, image_width=960, image_height=640, grid_w=30, grid_h=20, max_depth=80.0, num_depth_bins=40):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.stride_x = image_width / grid_w
        self.stride_y = image_height / grid_h
        self.max_depth = max_depth
        self.num_depth_bins = num_depth_bins
        self.bin_size = max_depth / num_depth_bins

    def _render_gaussian(self, heatmap, center_x, center_y, sigma=1.0):
        radius = int(3 * sigma)
        for y in range(max(0, center_y - radius), min(self.grid_h, center_y + radius + 1)):
            for x in range(max(0, center_x - radius), min(self.grid_w, center_x + radius + 1)):
                dist = (x - center_x)**2 + (y - center_y)**2
                val = math.exp(-dist / (2 * (sigma**2)))
                heatmap[y, x] = max(heatmap[y, x], val)
        return heatmap

    def encode(self, bboxes, num_valid_boxes):
        batch_size = bboxes.shape[0]
        target_cls = torch.zeros((batch_size, 3, self.grid_h, self.grid_w))
        target_loc = torch.zeros((batch_size, 2, self.grid_h, self.grid_w))
        target_depth_bin = torch.zeros((batch_size, self.grid_h, self.grid_w), dtype=torch.long)
        target_depth_res = torch.zeros((batch_size, 1, self.grid_h, self.grid_w))
        target_dim = torch.zeros((batch_size, 3, self.grid_h, self.grid_w))
        target_ori = torch.zeros((batch_size, 2, self.grid_h, self.grid_w))
        mask = torch.zeros((batch_size, 1, self.grid_h, self.grid_w))
        
        for b in range(batch_size):
            valid_count = num_valid_boxes[b].item()
            for i in range(valid_count):
                box = bboxes[b, i]
                cls_id = int(box[0].item()) - 1  
                
                pixel_x, pixel_y = box[1].item(), box[2].item()
                cx, cy, cz = box[3].item(), box[4].item(), box[5].item()
                length, width, height = box[6].item(), box[7].item(), box[8].item()
                heading = box[9].item()
                
                grid_x = int(pixel_x / self.stride_x)
                grid_y = int(pixel_y / self.stride_y)
                
                if 0 <= grid_x < self.grid_w and 0 <= grid_y < self.grid_h and 0 <= cls_id < 3:
                    mask[b, 0, grid_y, grid_x] = 1.0
                    target_cls[b, cls_id] = self._render_gaussian(target_cls[b, cls_id], grid_x, grid_y, sigma=1.0)
                    target_loc[b, 0, grid_y, grid_x] = (pixel_x / self.stride_x) - grid_x
                    target_loc[b, 1, grid_y, grid_x] = (pixel_y / self.stride_y) - grid_y
                    
                    depth_clamped = min(max(cx, 0.0), self.max_depth - 1e-4)
                    bin_idx = int(depth_clamped / self.bin_size)
                    residual = (depth_clamped - (bin_idx * self.bin_size)) / self.bin_size
                    
                    target_depth_bin[b, grid_y, grid_x] = bin_idx
                    target_depth_res[b, 0, grid_y, grid_x] = residual
                    target_dim[b, 0, grid_y, grid_x] = length
                    target_dim[b, 1, grid_y, grid_x] = width
                    target_dim[b, 2, grid_y, grid_x] = height
                    target_ori[b, 0, grid_y, grid_x] = math.sin(heading)
                    target_ori[b, 1, grid_y, grid_x] = math.cos(heading)
                    
        return {
            'class': target_cls,
            'location': target_loc,
            'depth_bin': target_depth_bin,
            'depth_res': target_depth_res,
            'dimensions': target_dim,
            'orientation': target_ori,
            'mask': mask
        }

class BEVGridEncoder:
    def __init__(self, x_range=(0.0, 80.0), y_range=(-40.0, 40.0), resolution=0.5):
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        self.grid_w = int((y_range[1] - y_range[0]) / resolution)
        self.grid_h = int((x_range[1] - x_range[0]) / resolution)

    def encode(self, bboxes, num_valid_boxes):
        batch_size = bboxes.shape[0]
        bev_grids = np.zeros((batch_size, 1, self.grid_h, self.grid_w), dtype=np.float32)
        
        for b in range(batch_size):
            valid_count = num_valid_boxes[b].item()
            for i in range(valid_count):
                box = bboxes[b, i]
                x, y = box[3].item(), box[4].item()
                length, width = box[6].item(), box[7].item()
                heading = box[9].item()
                
                if not (self.x_range[0] <= x <= self.x_range[1] and self.y_range[0] <= y <= self.y_range[1]):
                    continue
                    
                cos_a, sin_a = math.cos(heading), math.sin(heading)
                hl, hw = length / 2.0, width / 2.0
                corners = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
                rotation_matrix = np.array([[cos_a, -sin_a], [sin_a,  cos_a]])
                rotated_corners = (corners @ rotation_matrix.T) + np.array([x, y])
                
                grid_corners = np.zeros_like(rotated_corners)
                
                # Removed the inversion to match the CNN's natural spatial projection
                grid_corners[:, 0] = (rotated_corners[:, 1] - self.y_range[0]) / self.resolution
                
                # Map physical X (forward) to pixel Y (rows)
                grid_corners[:, 1] = self.grid_h - ((rotated_corners[:, 0] - self.x_range[0]) / self.resolution)
                
                grid_corners = grid_corners.astype(np.int32)
                cv2.fillConvexPoly(bev_grids[b, 0], grid_corners, 1.0)
                
        return {'bev_occupancy': torch.from_numpy(bev_grids)}