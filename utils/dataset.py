import os
import struct
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class WaymoDataset(Dataset):
    def __init__(self, tfrecord_path, max_boxes=100):
        self.tfrecord_path = tfrecord_path
        self.max_boxes = max_boxes 
        
        # Prevent TensorFlow from allocating GPU VRAM inside PyTorch workers
        tf.config.set_visible_devices([], 'GPU')
        
        # Build a lightweight byte-offset index for fast seeking without RAM bloat
        self.record_offsets = self._index_tfrecord(self.tfrecord_path)
        self.num_frames = len(self.record_offsets)

    def _index_tfrecord(self, file_path):
        """
        Fast binary scanner to map byte locations of each record.
        Uses ~0 MB of RAM regardless of dataset size.
        """
        offsets = []
        with open(file_path, 'rb') as f:
            while True:
                offset = f.tell()
                header = f.read(8)
                if not header or len(header) < 8:
                    break
                length = struct.unpack('<Q', header)[0]
                data_offset = offset + 12
                offsets.append((data_offset, length))
                # Skip to the start of the next record (header 8 + crc 4 + data length + crc 4 = 16 + length)
                f.seek(offset + 16 + length)
        return offsets

    def __len__(self):
        return self.num_frames
        
    def _extract_top_lidar_points(self, frame):
        # 1. Parse range images
        (range_images, camera_projections, 
         seg_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
        
        top_laser = open_dataset.LaserName.TOP
        
        # 2. CPU OPTIMIZATION TRICK: Remove non-TOP calibrations from the frame.
        # This forces the C++ Waymo utility to completely skip the heavy 
        # trigonometric math for the side, front, and rear lasers.
        calibrations = [c for c in frame.context.laser_calibrations if c.name == top_laser]
        del frame.context.laser_calibrations[:]
        frame.context.laser_calibrations.extend(calibrations)
        
        # 3. Safely run the point cloud conversion (it will now only process TOP)
        points, _ = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose
        )
        
        # FIX: Because we stripped the metadata down to 1 laser, 
        # the 'points' list now only contains exactly 1 element at index 0.
        point_cloud = points[0]
        
        # Filter points in front of ego-vehicle (X > 0)
        front_fov_mask = point_cloud[:, 0] > 0.0
        filtered_points = point_cloud[front_fov_mask]
        
        return torch.tensor(filtered_points, dtype=torch.float32)

    def _project_to_depth_map(self, lidar_points):
        IMAGE_WIDTH = 960
        IMAGE_HEIGHT = 640
        FOCAL_LENGTH = 1000.0
        CAMERA_HEIGHT_OFFSET = 1.5 
        
        x = lidar_points[:, 0]
        y = lidar_points[:, 1]
        z = lidar_points[:, 2]
        
        u = (IMAGE_WIDTH / 2) - (FOCAL_LENGTH * y / x)
        v = (IMAGE_HEIGHT / 2) - (FOCAL_LENGTH * (z - CAMERA_HEIGHT_OFFSET) / x)
        
        u = torch.round(u).long()
        v = torch.round(v).long()
        
        valid_mask = (u >= 0) & (u < IMAGE_WIDTH) & (v >= 0) & (v < IMAGE_HEIGHT)
        
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = x[valid_mask] 
        
        depth_map = torch.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.float32)
        
        sorted_indices = torch.argsort(depth_valid, descending=True)
        u_sorted = u_valid[sorted_indices]
        v_sorted = v_valid[sorted_indices]
        depth_sorted = depth_valid[sorted_indices]
        
        depth_map[0, v_sorted, u_sorted] = depth_sorted
        
        return depth_map

    def __getitem__(self, idx):
        # Seek directly to the byte location in the file
        data_offset, data_len = self.record_offsets[idx]
        with open(self.tfrecord_path, 'rb') as f:
            f.seek(data_offset)
            raw_data = f.read(data_len)
            
        frame = open_dataset.Frame()
        frame.ParseFromString(raw_data)
        
        # 1. RGB Extraction
        front_image_tensor = None
        for camera_image in frame.images:
            if camera_image.name == open_dataset.CameraName.FRONT:
                decoded_img = tf.io.decode_jpeg(camera_image.image).numpy()
                front_image_tensor = torch.from_numpy(decoded_img).permute(2, 0, 1)
                front_image_tensor = TF.resize(front_image_tensor, [640, 960], antialias=True)
                break 
                
        # 2. Bounding Box Extraction
        bboxes = np.zeros((self.max_boxes, 10), dtype=np.float32)
        valid_idx = 0
        IMAGE_WIDTH, IMAGE_HEIGHT = 960.0, 640.0
        FOCAL_LENGTH, CAMERA_HEIGHT_OFFSET = 1000.0, 1.5 
        
        for label in frame.laser_labels:
            if valid_idx >= self.max_boxes:
                break
                
            x, y, z = label.box.center_x, label.box.center_y, label.box.center_z
            heading = label.box.heading
            
            if x > 2.0 and abs(y / x) < 0.6: 
                pixel_x = (IMAGE_WIDTH / 2) - (FOCAL_LENGTH * y / x)
                pixel_y = (IMAGE_HEIGHT / 2) - (FOCAL_LENGTH * (z - CAMERA_HEIGHT_OFFSET) / x)
                
                if 0 <= pixel_x < IMAGE_WIDTH and 0 <= pixel_y < IMAGE_HEIGHT:
                    bboxes[valid_idx] = [
                        label.type, pixel_x, pixel_y, x, y, z,
                        label.box.length, label.box.width, label.box.height, heading
                    ]
                    valid_idx += 1

        # 3. Optimized TOP LiDAR Extraction & Depth Projection
        lidar_points = self._extract_top_lidar_points(frame)
        depth_map = self._project_to_depth_map(lidar_points)
        
        # 4. Augmentation
        is_train = 'train' in self.tfrecord_path
        if is_train:
            front_image_tensor = front_image_tensor.float() / 255.0
            jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            if random.random() > 0.5:
                front_image_tensor = jitter(front_image_tensor)
            front_image_tensor = front_image_tensor * 255.0
            
            if random.random() > 0.5:
                front_image_tensor = TF.hflip(front_image_tensor)
                depth_map = TF.hflip(depth_map)
                for i in range(valid_idx):
                    bboxes[i, 4] = -bboxes[i, 4] 
                    bboxes[i, 9] = -bboxes[i, 9] 
                    bboxes[i, 1] = 960.0 - bboxes[i, 1]

        rgbd_image_tensor = torch.cat((front_image_tensor, depth_map), dim=0)

        return {
            'timestamp': torch.tensor(frame.timestamp_micros, dtype=torch.int64),
            'front_image': rgbd_image_tensor,
            'bboxes': torch.from_numpy(bboxes),
            'num_valid_boxes': torch.tensor(valid_idx, dtype=torch.int32),
            'lidar_points': lidar_points
        }