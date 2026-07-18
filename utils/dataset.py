import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

class WaymoDataset(Dataset):
    def __init__(self, tfrecord_path, max_boxes=100):
        self.tfrecord_path = tfrecord_path
        self.max_boxes = max_boxes # The maximum number of labels we allow per frame
        self.raw_dataset = tf.data.TFRecordDataset(self.tfrecord_path, compression_type='')
        
        print("Initializing dataset and counting frames...")
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
                break 
                
        # 2. Extract 3D Bounding Boxes (LiDAR Labels)
        # Shape: [max_boxes, 8] -> [Class, X, Y, Z, Length, Width, Height, Heading]
        bboxes = np.zeros((self.max_boxes, 8), dtype=np.float32)
        
        num_valid_boxes = min(len(frame.laser_labels), self.max_boxes)
        
        for i in range(num_valid_boxes):
            label = frame.laser_labels[i]
            
            bboxes[i, 0] = label.type  # e.g., 1 for Vehicle, 2 for Pedestrian
            bboxes[i, 1] = label.box.center_x
            bboxes[i, 2] = label.box.center_y
            bboxes[i, 3] = label.box.center_z
            bboxes[i, 4] = label.box.length
            bboxes[i, 5] = label.box.width
            bboxes[i, 6] = label.box.height
            bboxes[i, 7] = label.box.heading

        return {
            'timestamp': torch.tensor(frame.timestamp_micros, dtype=torch.int64),
            'front_image': front_image_tensor,
            'bboxes': torch.from_numpy(bboxes),
            'num_valid_boxes': torch.tensor(num_valid_boxes, dtype=torch.int32)
        }

# --- Testing the Class ---
if __name__ == '__main__':
    data_path = 'data/raw/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
    
    waymo_data = WaymoDataset(data_path)
    dataloader = DataLoader(waymo_data, batch_size=4, shuffle=False)
    
    for batch in dataloader:
        print("\n--- Batch Extracted ---")
        print(f"Front Image Batch Shape: {batch['front_image'].shape}")
        
        # Check the new bounding box shapes
        bbox_shape = batch['bboxes'].shape
        print(f"Bounding Box Batch Shape: {bbox_shape}")
        print(f"Dimensions: [Batch_Size, Max_Boxes, Box_Attributes (Class + 7D Coords)]")
        print(f"Valid boxes per frame in this batch: {batch['num_valid_boxes'].tolist()}")
        break