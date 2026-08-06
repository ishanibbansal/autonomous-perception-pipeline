import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import math
import numpy as np

class BEVGridEncoder:
    def __init__(self, x_range=(0.0, 70.0), y_range=(-40.0, 40.0), bev_h=160, bev_w=160):
        self.x_range = x_range
        self.y_range = y_range
        self.grid_h = bev_h
        self.grid_w = bev_w
        self.res_x = (x_range[1] - x_range[0]) / bev_h  
        self.res_y = (y_range[1] - y_range[0]) / bev_w  

    def _render_gaussian(self, heatmap, center_x, center_y, sigma=1.0):
        radius = int(3 * sigma)
        y_min = max(0, center_y - radius)
        y_max = min(self.grid_h, center_y + radius + 1)
        x_min = max(0, center_x - radius)
        x_max = min(self.grid_w, center_x + radius + 1)
        
        if y_min >= y_max or x_min >= x_max:
            return heatmap
            
        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        dist = (xx - center_x)**2 + (yy - center_y)**2
        val = np.exp(-dist / (2 * (sigma**2)))
        heatmap[y_min:y_max, x_min:x_max] = np.maximum(heatmap[y_min:y_max, x_min:x_max], val)
        return heatmap

    def encode(self, bboxes, num_valid_boxes):
        batch_size = bboxes.shape[0]
        bev_grids = np.zeros((batch_size, 1, self.grid_h, self.grid_w), dtype=np.float32)
        
        dim_grids = np.zeros((batch_size, 3, self.grid_h, self.grid_w), dtype=np.float32)
        ori_grids = np.zeros((batch_size, 2, self.grid_h, self.grid_w), dtype=np.float32)
        # --- NEW: Offset Grid ---
        offset_grids = np.zeros((batch_size, 2, self.grid_h, self.grid_w), dtype=np.float32)
        mask_grids = np.zeros((batch_size, 1, self.grid_h, self.grid_w), dtype=np.float32)
        
        for b in range(batch_size):
            valid_count = num_valid_boxes[b].item()
            for i in range(valid_count):
                box = bboxes[b, i]
                
                x, y = box[3].item(), box[4].item()
                length, width, height = box[6].item(), box[7].item(), box[8].item()
                heading = box[9].item()
                
                if not (self.x_range[0] <= x <= self.x_range[1] and self.y_range[0] <= y <= self.y_range[1]):
                    continue
                    
                # 1. Calculate continuous floating-point grid coordinates
                ctx_feat = (y - self.y_range[0]) / self.res_y
                cty_feat = self.grid_h - 1 - ((x - self.x_range[0]) / self.res_x)
                
                # 2. Get discrete integer grid indices
                grid_cx = int(ctx_feat)
                grid_cy = int(cty_feat)
                
                # 3. Calculate fractional sub-pixel offset (The missing 0.0 to 1.0 remainder)
                offset_x = ctx_feat - grid_cx
                offset_y = cty_feat - grid_cy
                
                # Clamp boundaries safely
                grid_cx = max(0, min(grid_cx, self.grid_w - 1))
                grid_cy = max(0, min(grid_cy, self.grid_h - 1))
                
                radius_x = max(1.5, (length / self.res_x) / 2.0)
                radius_y = max(1.5, (width / self.res_y) / 2.0)
                sigma = max(1.5, max(radius_x, radius_y) / 2.0) 
                
                bev_grids[b, 0] = self._render_gaussian(bev_grids[b, 0], grid_cx, grid_cy, sigma=sigma)
                
                radius = int(sigma)
                y_min = max(0, grid_cy - radius)
                y_max = min(self.grid_h, grid_cy + radius + 1)
                x_min = max(0, grid_cx - radius)
                x_max = min(self.grid_w, grid_cx + radius + 1)
                
                dim_grids[b, 0, y_min:y_max, x_min:x_max] = length
                dim_grids[b, 1, y_min:y_max, x_min:x_max] = width
                dim_grids[b, 2, y_min:y_max, x_min:x_max] = height
                
                ori_grids[b, 0, y_min:y_max, x_min:x_max] = math.sin(heading)
                ori_grids[b, 1, y_min:y_max, x_min:x_max] = math.cos(heading)
                
                # --- NEW: Assign offset targets ---
                offset_grids[b, 0, y_min:y_max, x_min:x_max] = offset_x
                offset_grids[b, 1, y_min:y_max, x_min:x_max] = offset_y
                
                mask_grids[b, 0, y_min:y_max, x_min:x_max] = 1.0 
                
        return {
            'bev_occupancy': torch.from_numpy(bev_grids),
            'dimensions': torch.from_numpy(dim_grids),
            'orientation': torch.from_numpy(ori_grids),
            'offset': torch.from_numpy(offset_grids),
            'mask': torch.from_numpy(mask_grids)
        }