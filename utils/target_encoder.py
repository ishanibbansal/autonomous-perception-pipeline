import torch
import math

class TargetEncoder:
    def __init__(self, image_width=960, image_height=640, grid_w=30, grid_h=20, max_depth=80.0, num_depth_bins=40):
        """
        Maps raw 3D bounding box lists into dense spatial grids with CenterNet-style 
        heatmaps and bin-based depth targets.
        """
        self.grid_w = grid_w
        self.grid_h = grid_h
        
        self.stride_x = image_width / grid_w
        self.stride_y = image_height / grid_h
        
        self.max_depth = max_depth
        self.num_depth_bins = num_depth_bins
        self.bin_size = max_depth / num_depth_bins

    def _render_gaussian(self, heatmap, center_x, center_y, sigma=1.0):
        """Applies a 2D Gaussian splat to the heatmap for CenterNet-style classification."""
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
        target_loc = torch.zeros((batch_size, 2, self.grid_h, self.grid_w)) # Only X, Y offsets
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
                    
                    # 1. Heatmap Classification
                    target_cls[b, cls_id] = self._render_gaussian(target_cls[b, cls_id], grid_x, grid_y, sigma=1.0)
                    
                    # 2. 2D Center Offsets
                    offset_x = (pixel_x / self.stride_x) - grid_x
                    offset_y = (pixel_y / self.stride_y) - grid_y
                    target_loc[b, 0, grid_y, grid_x] = offset_x
                    target_loc[b, 1, grid_y, grid_x] = offset_y
                    
                    # 3. Bin-Based Depth Encoding
                    depth_clamped = min(max(cx, 0.0), self.max_depth - 1e-4)
                    bin_idx = int(depth_clamped / self.bin_size)
                    residual = (depth_clamped - (bin_idx * self.bin_size)) / self.bin_size
                    
                    target_depth_bin[b, grid_y, grid_x] = bin_idx
                    target_depth_res[b, 0, grid_y, grid_x] = residual
                    
                    # 4. Dimensions & Orientation
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

if __name__ == '__main__':
    from dataset import WaymoDataset
    from torch.utils.data import DataLoader

    print("Initializing TargetEncoder Test...")
    data_path = 'data/raw/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
    
    try:
        waymo_data = WaymoDataset(data_path)
        dataloader = DataLoader(waymo_data, batch_size=4, shuffle=False)
        batch = next(iter(dataloader))
        
        encoder = TargetEncoder(image_width=960, image_height=640, grid_w=30, grid_h=20)
        targets = encoder.encode(batch['bboxes'], batch['num_valid_boxes'])
        
        print("\n--- Encoded Target Shapes ---")
        for key, tensor in targets.items():
            print(f"{key.capitalize():<12}: {tensor.shape}")
            
        total_objects_in_batch = batch['num_valid_boxes'].sum().item()
        total_cells_activated = targets['mask'].sum().item()
        
        print("\n--- Validation ---")
        print(f"Raw valid boxes in batch : {total_objects_in_batch}")
        print(f"Grid anchors activated   : {int(total_cells_activated)}")
        print("SUCCESS: TargetEncoder upgraded to Bin-Based Depth + Heatmaps!")
            
    except FileNotFoundError:
        print(f"Test failed: Could not find the dataset at {data_path}.")