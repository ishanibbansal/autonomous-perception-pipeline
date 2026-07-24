import torch
import math

class TargetEncoder:
    def __init__(self, image_width=1920, image_height=1280, grid_w=60, grid_h=40):
        """
        Maps raw bounding box lists into dense spatial grids for loss calculation.
        """
        self.grid_w = grid_w
        self.grid_h = grid_h
        
        # Calculate the downsampling stride (e.g., 1920 / 60 = 32)
        self.stride_x = image_width / grid_w
        self.stride_y = image_height / grid_h

    def encode(self, bboxes, num_valid_boxes):
        """
        Args:
            bboxes: Tensor of shape [Batch, Max_Boxes, 10]
                    10 dims -> [Class, Pix_X, Pix_Y, 3D_X, 3D_Y, 3D_Z, Length, Width, Height, Heading]
            num_valid_boxes: Tensor of shape [Batch] indicating how many boxes are real
        Returns:
            Dictionary of target tensors matching the model's output shapes.
        """
        batch_size = bboxes.shape[0]
        
        # Initialize empty grids with zeros
        target_cls = torch.zeros((batch_size, 3, self.grid_h, self.grid_w))
        target_loc = torch.zeros((batch_size, 3, self.grid_h, self.grid_w))
        target_dim = torch.zeros((batch_size, 3, self.grid_h, self.grid_w))
        target_ori = torch.zeros((batch_size, 2, self.grid_h, self.grid_w))
        
        # Mask to tell the loss function which cells actually contain objects
        mask = torch.zeros((batch_size, 1, self.grid_h, self.grid_w))
        
        for b in range(batch_size):
            valid_count = num_valid_boxes[b].item()
            
            for i in range(valid_count):
                box = bboxes[b, i]
                cls_id = int(box[0].item()) - 1  # Shift 1-based class ID to 0-based index
                
                # Extract coordinates from the new 10D tensor
                pixel_x, pixel_y = box[1].item(), box[2].item()
                cx, cy, cz = box[3].item(), box[4].item(), box[5].item()
                length, width, height = box[6].item(), box[7].item(), box[8].item()
                heading = box[9].item()
                
                # 1. Map continuous image PIXELS to discrete grid indices
                grid_x = int(pixel_x / self.stride_x)
                grid_y = int(pixel_y / self.stride_y)
                
                # Ensure the projected center falls within our 40x60 grid and valid classes
                if 0 <= grid_x < self.grid_w and 0 <= grid_y < self.grid_h and 0 <= cls_id < 3:
                    
                    # Mark the cell as containing an object
                    mask[b, 0, grid_y, grid_x] = 1.0
                    
                    # 2. Classification: Set probability to 1.0 for the correct class layer
                    target_cls[b, cls_id, grid_y, grid_x] = 1.0
                    
                    # 3. Location: Assign the raw 3D meters directly
                    target_loc[b, 0, grid_y, grid_x] = cx
                    target_loc[b, 1, grid_y, grid_x] = cy
                    target_loc[b, 2, grid_y, grid_x] = cz
                    
                    # 4. Dimensions: Assign raw metric scale
                    target_dim[b, 0, grid_y, grid_x] = length
                    target_dim[b, 1, grid_y, grid_x] = width
                    target_dim[b, 2, grid_y, grid_x] = height
                    
                    # 5. Orientation: Convert heading angle (radians) to Sine/Cosine pairs
                    target_ori[b, 0, grid_y, grid_x] = math.sin(heading)
                    target_ori[b, 1, grid_y, grid_x] = math.cos(heading)
                    
        return {
            'class': target_cls,
            'location': target_loc,
            'dimensions': target_dim,
            'orientation': target_ori,
            'mask': mask
        }

# --- Testing the Encoder ---
if __name__ == '__main__':
    from dataset import WaymoDataset
    from torch.utils.data import DataLoader

    print("Initializing TargetEncoder Test...")
    
    # Use the same raw tfrecord file tested in dataset.py
    data_path = 'data/raw/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
    
    try:
        waymo_data = WaymoDataset(data_path)
        dataloader = DataLoader(waymo_data, batch_size=4, shuffle=False)
        
        # Grab exactly one batch
        batch = next(iter(dataloader))
        
        encoder = TargetEncoder()
        targets = encoder.encode(batch['bboxes'], batch['num_valid_boxes'])
        
        print("\n--- Encoded Target Shapes ---")
        for key, tensor in targets.items():
            print(f"{key.capitalize():<12}: {tensor.shape}")
            
        # Verify the math by checking how many grid cells were activated
        total_objects_in_batch = batch['num_valid_boxes'].sum().item()
        total_cells_activated = targets['mask'].sum().item()
        
        print("\n--- Validation ---")
        print(f"Raw valid boxes in batch : {total_objects_in_batch}")
        print(f"Grid cells activated     : {int(total_cells_activated)}")
        
        if total_cells_activated > 0:
            print("SUCCESS: The encoder successfully projected raw bounding boxes onto the 40x60 grid!")
        else:
            print("WARNING: No grid cells were activated. Check coordinate scaling.")
            
    except FileNotFoundError:
        print(f"Test failed: Could not find the dataset at {data_path}. Make sure you run this from the correct directory.")