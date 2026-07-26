import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class WaymoDataset(Dataset):
    def __init__(self, tfrecord_path, max_boxes=100):
        self.tfrecord_path = tfrecord_path
        self.max_boxes = max_boxes 
        self.raw_dataset = tf.data.TFRecordDataset(self.tfrecord_path, compression_type='')
        
        print(f"Initializing dataset and counting frames for {os.path.basename(tfrecord_path)}...")
        self.frame_list = [data.numpy() for data in self.raw_dataset]
        self.num_frames = len(self.frame_list)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        raw_data = self.frame_list[idx]
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(raw_data))
        
        # 1. Extract Front Camera Image
        front_image_tensor = None
        for camera_image in frame.images:
            if camera_image.name == open_dataset.CameraName.FRONT:
                decoded_img = tf.io.decode_jpeg(camera_image.image).numpy()
                front_image_tensor = torch.from_numpy(decoded_img).permute(2, 0, 1)
                
                # ---> DOWNSCALE IMAGE 50% <---
                front_image_tensor = TF.resize(front_image_tensor, [640, 960], antialias=True)
                break 
                
        # ---> AUGMENTATION BLOCK <---
        is_train = 'train' in self.tfrecord_path
        do_hflip = is_train and random.random() > 0.5
        
        if is_train:
            # Photometric: Randomize lighting, contrast, and saturation
            jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
            front_image_tensor = jitter(front_image_tensor)
            
        if do_hflip:
            # Spatial: Flip the image tensor horizontally
            front_image_tensor = TF.hflip(front_image_tensor)
        # ----------------------------
                
        # 2. Extract 3D Bounding Boxes and Project geometrically to 2D
        bboxes = np.zeros((self.max_boxes, 10), dtype=np.float32)
        valid_idx = 0
        
        # ---> SCALED CAMERA INTRINSICS (50%) <---
        IMAGE_WIDTH = 960.0    # Was 1920
        IMAGE_HEIGHT = 640.0   # Was 1280
        FOCAL_LENGTH = 1000.0  # Was 2000.0
        CAMERA_HEIGHT_OFFSET = 1.5 
        
        for label in frame.laser_labels:
            if valid_idx >= self.max_boxes:
                break
                
            x = label.box.center_x # Depth (Forward)
            y = label.box.center_y # Horizontal (Left)
            z = label.box.center_z # Vertical (Up)
            heading = label.box.heading
            
            # ---> APPLY SPATIAL FLIP TO 3D TARGETS <---
            if do_hflip:
                y = -y
                heading = -heading
            # ------------------------------------------
            
            # Step 1: Physical FOV Filter 
            if x > 2.0 and abs(y / x) < 0.6: 
                
                # Step 2: Geometric Projection
                pixel_x = (IMAGE_WIDTH / 2) - (FOCAL_LENGTH * y / x)
                pixel_y = (IMAGE_HEIGHT / 2) - (FOCAL_LENGTH * (z - CAMERA_HEIGHT_OFFSET) / x)
                
                # Step 3: Validate the projection lands inside the 960x640 image frame
                if 0 <= pixel_x < IMAGE_WIDTH and 0 <= pixel_y < IMAGE_HEIGHT:
                    bboxes[valid_idx, 0] = label.type  
                    bboxes[valid_idx, 1] = pixel_x
                    bboxes[valid_idx, 2] = pixel_y
                    bboxes[valid_idx, 3] = x
                    bboxes[valid_idx, 4] = y
                    bboxes[valid_idx, 5] = z
                    bboxes[valid_idx, 6] = label.box.length
                    bboxes[valid_idx, 7] = label.box.width
                    bboxes[valid_idx, 8] = label.box.height
                    bboxes[valid_idx, 9] = heading  
                    
                    valid_idx += 1

        return {
            'timestamp': torch.tensor(frame.timestamp_micros, dtype=torch.int64),
            'front_image': front_image_tensor,
            'bboxes': torch.from_numpy(bboxes),
            'num_valid_boxes': torch.tensor(valid_idx, dtype=torch.int32)
        }

# --- Testing the Class ---
if __name__ == '__main__':
    data_path = 'data/raw/train/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
    # Fallback in case testing file doesn't exist at the new nested path
    if not os.path.exists(data_path):
        import glob
        train_files = glob.glob('data/raw/train/*.tfrecord')
        if train_files:
            data_path = train_files[0]
            
    waymo_data = WaymoDataset(data_path)
    dataloader = DataLoader(waymo_data, batch_size=4, shuffle=False)
    
    for batch in dataloader:
        print("\n--- Batch Extracted ---")
        print(f"Front Image Batch Shape: {batch['front_image'].shape}")
        print(f"Bounding Box Batch Shape: {batch['bboxes'].shape}")
        print(f"Valid visible boxes per frame: {batch['num_valid_boxes'].tolist()}")
        break