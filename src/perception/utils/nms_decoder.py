import torch
import torch.nn.functional as F
import math

class CenterNetDecoder:
    def __init__(self, x_range=(0.0, 70.0), y_range=(-40.0, 40.0), bev_h=160, bev_w=160, threshold=0.35):
        self.x_range = x_range
        self.y_range = y_range
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.threshold = threshold
        self.res_x = (x_range[1] - x_range[0]) / bev_h
        self.res_y = (y_range[1] - y_range[0]) / bev_w
        
    def _radius_nms(self, boxes, min_radius=2.5):
        """
        Aggressive suppression: Removes any secondary box within 2.5 meters of a primary peak.
        Ensures strict single-box-per-vehicle decoding.
        """
        if not boxes:
            return []
            
        boxes = sorted(boxes, key=lambda b: b['score'], reverse=True)
        keep = []
        
        for box in boxes:
            is_duplicate = False
            for kept_box in keep:
                dist = math.hypot(box['x'] - kept_box['x'], box['y'] - kept_box['y'])
                if dist < min_radius:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                keep.append(box)
                
        return keep
        
    def decode(self, predictions):
        """
        Translates CenterNet tensor predictions into rigid physical bounding boxes.
        """
        occ_logits = torch.flip(predictions['bev_occupancy'], dims=[-1])
        dim_preds = torch.flip(predictions['dimensions'], dims=[-1])
        ori_preds = torch.flip(predictions['orientation'], dims=[-1])
        off_preds = torch.flip(predictions['offset'], dims=[-1])
        
        heatmap = torch.sigmoid(occ_logits)
        
        # 1. 5x5 Max Pooling (Creates a local suppression zone)
        max_pool = F.max_pool2d(heatmap, kernel_size=5, stride=1, padding=2)
        peak_mask = (heatmap == max_pool) & (heatmap > self.threshold)
        
        batch_size = heatmap.shape[0]
        batch_boxes = []
        
        for b in range(batch_size):
            # 2. Extract ALL peaks to prevent Top-K Starvation on saturated plateaus
            peaks = peak_mask[b, 0].nonzero(as_tuple=False)
            boxes = []
            
            for peak in peaks:
                grid_cy, grid_cx = peak[0].item(), peak[1].item()
                score = heatmap[b, 0, grid_cy, grid_cx].item()
                
                # [FIX 1]: Negate offset_x because the grid was flipped horizontally
                offset_x = -off_preds[b, 0, grid_cy, grid_cx].item()
                offset_y = off_preds[b, 1, grid_cy, grid_cx].item()
                
                c_x = grid_cx + offset_x
                c_y = grid_cy + offset_y
                
                y_physical = ( (self.bev_w - 1 - c_x) * self.res_y ) + self.y_range[0]
                x_physical = ( (self.bev_h - 1 - c_y) * self.res_x ) + self.x_range[0]
                
                length = dim_preds[b, 0, grid_cy, grid_cx].item()
                width = dim_preds[b, 1, grid_cy, grid_cx].item()
                height = dim_preds[b, 2, grid_cy, grid_cx].item()
                
                # [FIX 2]: Negate sin_h because the grid was flipped horizontally
                sin_h = -ori_preds[b, 0, grid_cy, grid_cx].item()
                cos_h = ori_preds[b, 1, grid_cy, grid_cx].item()
                heading = math.atan2(sin_h, cos_h)
                
                z_physical = height / 2.0 
                
                boxes.append({
                    'x': x_physical, 'y': y_physical, 'z': z_physical,
                    'length': length, 'width': width, 'height': height,
                    'heading': heading, 'score': score
                })
                
            # 3. Final Euclidean Sweep to collapse plateaus down to 1 box
            filtered_boxes = self._radius_nms(boxes, min_radius=2.5)
            batch_boxes.append(filtered_boxes)
            
        return batch_boxes