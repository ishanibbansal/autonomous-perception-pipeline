import torch
import math

class TargetEncoder:
    def __init__(self, image_width=960, image_height=640, grid_w=30, grid_h=20):
        """
        Maps raw bounding box lists into dense spatial grids for loss calculation.
        """
        self.grid_w = grid_w
        self.grid_h = grid_h
        
        self.stride_x = image_width / grid_w
        self.stride_y = image_height / grid_h

    def encode(self, bboxes, num_valid_boxes):
        batch_size = bboxes.shape[0]
        
        target_cls = torch.zeros((batch_size, 3, self.grid_h, self.grid_w))
        target_loc = torch.zeros((batch_size, 3, self.grid_h, self.grid_w))
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
                    target_cls[b, cls_id, grid_y, grid_x] = 1.0
                    
                    # 3. Location: Calculate relative pixel offset + absolute depth
                    offset_x = (pixel_x / self.stride_x) - grid_x
                    offset_y = (pixel_y / self.stride_y) - grid_y
                    
                    target_loc[b, 0, grid_y, grid_x] = offset_x
                    target_loc[b, 1, grid_y, grid_x] = offset_y
                    target_loc[b, 2, grid_y, grid_x] = cx # cx is the forward depth
                    
                    # Standard processing applied for dimensions and orientation
                    target_dim[b, 0, grid_y, grid_x] = length
                    target_dim[b, 1, grid_y, grid_x] = width
                    target_dim[b, 2, grid_y, grid_x] = height
                    
                    target_ori[b, 0, grid_y, grid_x] = math.sin(heading)
                    target_ori[b, 1, grid_y, grid_x] = math.cos(heading)
                    
        return {
            'class': target_cls,
            'location': target_loc,
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
        print(f"Grid cells activated     : {int(total_cells_activated)}")
        
        if total_cells_activated > 0:
            print("SUCCESS: The encoder successfully projected raw bounding boxes onto the 30x20 grid!")
        else:
            print("WARNING: No grid cells were activated. Check coordinate scaling.")
            
    except FileNotFoundError:
        print(f"Test failed: Could not find the dataset at {data_path}.")