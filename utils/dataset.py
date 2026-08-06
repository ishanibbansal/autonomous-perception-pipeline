import os
import struct
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
from PIL import Image
import io

class WaymoDataset(Dataset):
    def __init__(self, tfrecord_path, max_boxes=100, is_train=False, num_sweeps=3):
        self.tfrecord_path = tfrecord_path
        self.max_boxes = max_boxes 
        self.is_train = is_train
        self.num_sweeps = num_sweeps 
        
        self._frame_cache = OrderedDict()
        
        # --- FIX 1: Drastically reduce cache capacity ---
        # We only need enough history to satisfy num_sweeps, preventing massive RAM bloat.
        self._cache_capacity = max(self.num_sweeps + 2, 5)
        
        tf.config.set_visible_devices([], 'GPU')
        self.record_offsets = self._index_tfrecord(self.tfrecord_path)
        self.num_frames = len(self.record_offsets)

    def _index_tfrecord(self, file_path):
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
                f.seek(offset + 16 + length)
        return offsets

    def __len__(self):
        return self.num_frames
        
    def _get_frame(self, idx):
        if idx in self._frame_cache:
            self._frame_cache.move_to_end(idx)
            return self._frame_cache[idx]
            
        data_offset, data_len = self.record_offsets[idx]
        with open(self.tfrecord_path, 'rb') as f:
            f.seek(data_offset)
            raw_data = f.read(data_len)
        frame = open_dataset.Frame()
        frame.ParseFromString(raw_data)
        
        if len(self._frame_cache) >= self._cache_capacity:
            self._frame_cache.popitem(last=False)
        self._frame_cache[idx] = frame
        return frame

    def _extract_fusion_data(self, frame, is_current_frame=True):
        (range_images, camera_projections, 
         seg_labels, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
        
        top_laser = open_dataset.LaserName.TOP
        calibrations = [c for c in frame.context.laser_calibrations if c.name == top_laser]
        del frame.context.laser_calibrations[:]
        frame.context.laser_calibrations.extend(calibrations)
        
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose)
        
        point_cloud = points[0]
        point_cp = cp_points[0] 
        
        uvs = np.full((point_cloud.shape[0], 2), -9999.0, dtype=np.float32)
        
        if is_current_frame:
            front_cam_id = open_dataset.CameraName.FRONT
            front_cam_mask = (point_cp[:, 0] == front_cam_id)
            uvs[front_cam_mask] = point_cp[front_cam_mask, 1:3]
            
        return point_cloud, uvs

    def _extract_front_image(self, frame):
        for img in frame.images:
            if img.name == open_dataset.CameraName.FRONT:
                # --- FIX 2: Use PIL instead of tf.io inside the dataloader worker ---
                # This prevents TensorFlow memory allocators from leaking RAM in PyTorch sub-processes.
                pil_img = Image.open(io.BytesIO(img.image))
                decoded_img = np.array(pil_img)
                img_tensor = (decoded_img.astype(np.float32) / 255.0).transpose(2, 0, 1)
                return img_tensor
        return np.zeros((3, 640, 960), dtype=np.float32)

    def __getitem__(self, idx):
        current_frame = self._get_frame(idx)
        current_time = current_frame.timestamp_micros
        current_pose = np.reshape(np.array(current_frame.pose.transform), [4, 4])
        
        front_image = self._extract_front_image(current_frame)
        
        all_points = []
        all_uvs = []
        
        for sweep_idx in range(idx, max(-1, idx - self.num_sweeps), -1):
            frame = self._get_frame(sweep_idx)
            dt_sec = (frame.timestamp_micros - current_time) / 1e6
            if dt_sec < -1.0:
                break
                
            sweep_pose = np.reshape(np.array(frame.pose.transform), [4, 4])
            is_current = (sweep_idx == idx)
            
            lidar_points, lidar_uvs = self._extract_fusion_data(frame, is_current_frame=is_current)
            
            if not is_current:
                xyz = lidar_points[:, :3]
                xyz_homogeneous = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
                transform_matrix = np.linalg.inv(current_pose) @ sweep_pose
                aligned_xyz = (transform_matrix @ xyz_homogeneous.T).T[:, :3]
                lidar_points[:, :3] = aligned_xyz
                
            dt_feature = np.full((lidar_points.shape[0], 1), dt_sec, dtype=np.float32)
            lidar_points = np.concatenate([lidar_points, dt_feature], axis=1)
            
            front_mask = lidar_points[:, 0] > 0.0
            all_points.append(lidar_points[front_mask])
            all_uvs.append(lidar_uvs[front_mask])

        fused_lidar_points = np.concatenate(all_points, axis=0)
        fused_lidar_uvs = np.concatenate(all_uvs, axis=0)
                
        bboxes = np.zeros((self.max_boxes, 10), dtype=np.float32)
        valid_idx = 0
        
        for label in current_frame.laser_labels:
            if valid_idx >= self.max_boxes:
                break
            x, y, z = label.box.center_x, label.box.center_y, label.box.center_z
            l, w, h = label.box.length, label.box.width, label.box.height
            heading = label.box.heading
            
            if x > 0.0:
                box_array = np.array([label.type, 0.0, 0.0, x, y, z, l, w, h, heading], dtype=np.float32)
                bboxes[valid_idx] = box_array
                valid_idx += 1

        if self.is_train and valid_idx > 0:
            scale = random.uniform(0.95, 1.05)
            fused_lidar_points[:, :3] *= scale
            bboxes[:valid_idx, 3:6] *= scale
            bboxes[:valid_idx, 6:9] *= scale
            
            if random.random() > 0.5:
                # 1. Flip 3D LiDAR & BBoxes
                fused_lidar_points[:, 1] = -fused_lidar_points[:, 1] 
                bboxes[:valid_idx, 4] = -bboxes[:valid_idx, 4]
                bboxes[:valid_idx, 9] = -bboxes[:valid_idx, 9]
                
                # 2. Flip 2D Front Camera Image (Shape: [3, 640, 960])
                front_image = np.ascontiguousarray(np.flip(front_image, axis=2))
                
                # 3. Reflect UV coordinates across image width (960px)
                valid_uv_mask = fused_lidar_uvs[:, 0] != -9999.0
                fused_lidar_uvs[valid_uv_mask, 0] = 960.0 - fused_lidar_uvs[valid_uv_mask, 0]

        return {
            'timestamp': torch.tensor(current_frame.timestamp_micros, dtype=torch.int64),
            'camera_image': torch.from_numpy(front_image),
            'bboxes': torch.from_numpy(bboxes),
            'num_valid_boxes': torch.tensor(valid_idx, dtype=torch.int32),
            'lidar_points': torch.from_numpy(fused_lidar_points),
            'lidar_uvs': torch.from_numpy(fused_lidar_uvs)
        }

def waymo_collate_fn(batch):
    timestamps = []
    camera_images = []
    bboxes = []
    num_valid_boxes = []
    all_points = []
    all_uvs = []
    batch_indices = []

    for i, item in enumerate(batch):
        timestamps.append(item['timestamp'])
        camera_images.append(item['camera_image'])
        bboxes.append(item['bboxes'])
        num_valid_boxes.append(item['num_valid_boxes'])
        
        points = item['lidar_points']
        uvs = item['lidar_uvs']
        all_points.append(points)
        all_uvs.append(uvs)
        
        batch_indices.append(torch.full((points.shape[0],), i, dtype=torch.long))

    return {
        'timestamp': torch.stack(timestamps),
        'camera_images': torch.stack(camera_images),
        'bboxes': torch.stack(bboxes),
        'num_valid_boxes': torch.stack(num_valid_boxes),
        'lidar_points': torch.cat(all_points, dim=0),
        'lidar_uvs': torch.cat(all_uvs, dim=0),
        'batch_indices': torch.cat(batch_indices, dim=0)
    }