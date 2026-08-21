import os
import struct
import random
import numpy as np
import torch
import torchvision.transforms as T
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
        self._cache_capacity = self.num_sweeps + 1
        
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
                pil_img = Image.open(io.BytesIO(img.image))
                decoded_img = np.array(pil_img)
                img_tensor = (decoded_img.astype(np.float32) / 255.0).transpose(2, 0, 1)
                return img_tensor
        # [FIX]: Match the actual 1280x1920 Waymo resolution fallback
        return np.zeros((3, 1280, 1920), dtype=np.float32)

    def _extract_camera_params(self, frame):
        for calib in frame.context.camera_calibrations:
            if calib.name == open_dataset.CameraName.FRONT:
                f_u, f_v, c_u, c_v = calib.intrinsic[:4]
                intrinsics = np.array([
                    [f_u, 0.0, c_u],
                    [0.0, f_v, c_v],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float32)
                extrinsics = np.array(calib.extrinsic.transform, dtype=np.float32).reshape(4, 4)
                return intrinsics, extrinsics
        return np.eye(3, dtype=np.float32), np.eye(4, dtype=np.float32)

    def __getitem__(self, idx):
        current_frame = self._get_frame(idx)
        current_time = current_frame.timestamp_micros
        current_pose = np.reshape(np.array(current_frame.pose.transform), [4, 4])
        
        front_image = self._extract_front_image(current_frame)
        intrinsics, extrinsics = self._extract_camera_params(current_frame)
        
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

        # [FIX]: Generate 80x120 Depth Labels
        depth_label = np.full((80, 120), -1, dtype=np.int64) 
        
        # [FIX]: Use 1920 width / 1280 height boundaries
        valid_mask = (fused_lidar_uvs[:, 0] >= 0) & (fused_lidar_uvs[:, 0] < 1920) & \
                     (fused_lidar_uvs[:, 1] >= 0) & (fused_lidar_uvs[:, 1] < 1280) & \
                     (fused_lidar_points[:, 0] >= 2.0) & (fused_lidar_points[:, 0] < 50.0)
                     
        if valid_mask.any():
            v_uvs = fused_lidar_uvs[valid_mask]
            v_depths = fused_lidar_points[valid_mask, 0]
            
            sort_idx = np.argsort(v_depths)[::-1]
            v_uvs, v_depths = v_uvs[sort_idx], v_depths[sort_idx]
            
            # [FIX]: Clip to 119 and 79 max indices
            u_feat = np.clip((v_uvs[:, 0] / 16.0).astype(np.int32), 0, 119)
            v_feat = np.clip((v_uvs[:, 1] / 16.0).astype(np.int32), 0, 79)
            
            d_bins = np.clip(((v_depths - 2.0) / 1.0).astype(np.int64), 0, 47)
            depth_label[v_feat, u_feat] = d_bins
                
        bboxes = np.zeros((self.max_boxes, 10), dtype=np.float32)
        valid_idx = 0
        
        for label in current_frame.laser_labels:
            if valid_idx >= self.max_boxes:
                break
            x, y, z = label.box.center_x, label.box.center_y, label.box.center_z
            l, w, h = label.box.length, label.box.width, label.box.height
            heading = label.box.heading
            
            if x > 2.0 and abs(y / x) < 0.6:
                box_array = np.array([label.type, 0.0, 0.0, x, y, z, l, w, h, heading], dtype=np.float32)
                bboxes[valid_idx] = box_array
                valid_idx += 1

        if self.is_train and valid_idx > 0:
            scale = random.uniform(0.95, 1.05)
            fused_lidar_points[:, :3] *= scale
            bboxes[:valid_idx, 3:6] *= scale
            bboxes[:valid_idx, 6:9] *= scale
            
            if random.random() > 0.5:
                fused_lidar_points[:, 1] = -fused_lidar_points[:, 1] 
                bboxes[:valid_idx, 4] = -bboxes[:valid_idx, 4]
                bboxes[:valid_idx, 9] = -bboxes[:valid_idx, 9]
                
                front_image = np.ascontiguousarray(np.flip(front_image, axis=2))
                valid_uv_mask = fused_lidar_uvs[:, 0] != -9999.0
                
                # [FIX]: Invert horizontally across 1920 pixels
                fused_lidar_uvs[valid_uv_mask, 0] = 1920.0 - fused_lidar_uvs[valid_uv_mask, 0]
                intrinsics[0, 2] = 1920.0 - intrinsics[0, 2]
                depth_label = np.ascontiguousarray(np.flip(depth_label, axis=1))

        return {
            'timestamp': torch.tensor(current_frame.timestamp_micros, dtype=torch.int64),
            'camera_image': torch.from_numpy(front_image),       
            'intrinsics': torch.from_numpy(intrinsics),          
            'extrinsics': torch.from_numpy(extrinsics),           
            'depth_label': torch.from_numpy(depth_label), 
            'bboxes': torch.from_numpy(bboxes),
            'num_valid_boxes': torch.tensor(valid_idx, dtype=torch.int32),
            'lidar_points': torch.from_numpy(fused_lidar_points),
            'lidar_uvs': torch.from_numpy(fused_lidar_uvs)
        }

def waymo_collate_fn(batch):
    timestamps = []
    camera_images = []
    intrinsics = []
    extrinsics = []
    depth_labels = []
    bboxes = []
    num_valid_boxes = []
    all_points = []
    all_uvs = []
    batch_indices = []

    for i, item in enumerate(batch):
        timestamps.append(item['timestamp'])
        camera_images.append(item['camera_image'])
        intrinsics.append(item['intrinsics']) 
        extrinsics.append(item['extrinsics'])  
        depth_labels.append(item['depth_label']) 
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
        'intrinsics': torch.stack(intrinsics), 
        'extrinsics': torch.stack(extrinsics),  
        'depth_labels': torch.stack(depth_labels), 
        'bboxes': torch.stack(bboxes),
        'num_valid_boxes': torch.stack(num_valid_boxes),
        'lidar_points': torch.cat(all_points, dim=0),
        'lidar_uvs': torch.cat(all_uvs, dim=0),
        'batch_indices': torch.cat(batch_indices, dim=0)
    }

# ==========================================
# PHASE 2/3: TEMPORAL DATASET EXTENSIONS
# ==========================================

class WaymoTemporalDataset(WaymoDataset):
    """
    Extended dataset for Memory-Safe Temporal Fusion (History Caching).
    Returns a sequence of frames (t_0, t_1, t_2) instead of randomized single frames.
    Inherits from WaymoDataset to safely reuse TFRecord parsing utilities.
    """
    def __init__(self, tfrecord_path, max_boxes=100, is_train=False, seq_length=3):
        # Initialize the parent class but disable LiDAR sweeps to save I/O time
        super().__init__(tfrecord_path, max_boxes=max_boxes, is_train=is_train, num_sweeps=0)
        self.seq_length = seq_length
        
        # Expand the frame cache to comfortably hold the entire temporal sequence in RAM
        self._cache_capacity = self.seq_length * 2 

        # [NEW]: Initialize Photometric Augmentation for training
        self.color_jitter = T.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.2, 
            hue=0.05
        ) if self.is_train else None

    def __getitem__(self, idx):
        sequence_data = []
        
        # 1. Anchor Frame (t_0)
        anchor_frame = self._get_frame(idx)
        anchor_time = anchor_frame.timestamp_micros
        
        for step in range(self.seq_length):
            target_idx = max(0, idx - step)
            frame = self._get_frame(target_idx)
            
            # 2. Scene Boundary Safety Check
            # If the timestamp jumps significantly (e.g., end of one log, start of another),
            # we duplicate the oldest valid frame to prevent cross-city temporal contamination.
            dt_sec = (anchor_time - frame.timestamp_micros) / 1e6
            if dt_sec > (step * 0.5 + 1.0): 
                target_idx = max(0, idx - step + 1)
                frame = self._get_frame(target_idx)
                
            # 3. Extract Core Camera Inputs
            front_image = self._extract_front_image(frame)
            intrinsics, extrinsics = self._extract_camera_params(frame)
            
            # Convert NumPy image to Tensor early to apply augmentations
            img_tensor = torch.from_numpy(front_image)
            
            # [NEW]: Apply photometric jitter independently to each frame in the sequence
            if self.color_jitter is not None:
                img_tensor = self.color_jitter(img_tensor)
            
            # 4. Extract Targets ONLY for the current frame (t_0)
            if step == 0:
                bboxes = np.zeros((self.max_boxes, 10), dtype=np.float32)
                valid_idx = 0
                for label in frame.laser_labels:
                    if valid_idx >= self.max_boxes:
                        break
                    x, y, z = label.box.center_x, label.box.center_y, label.box.center_z
                    l, w, h = label.box.length, label.box.width, label.box.height
                    heading = label.box.heading
                    
                    if x > 2.0 and abs(y / x) < 0.6:
                        bboxes[valid_idx] = np.array([label.type, 0.0, 0.0, x, y, z, l, w, h, heading], dtype=np.float32)
                        valid_idx += 1
                        
                # Extract single-frame LiDAR strictly to generate the depth labels
                lidar_points, lidar_uvs = self._extract_fusion_data(frame, is_current_frame=True)

                # [CRITICAL FIX]: Append the missing time-delta feature (dt_sec = 0.0)
                # The Teacher model was trained in Phase 1 where this column always existed.
                dt_feature = np.zeros((lidar_points.shape[0], 1), dtype=np.float32)
                lidar_points = np.concatenate([lidar_points, dt_feature], axis=1)
                depth_label = np.full((80, 120), -1, dtype=np.int64) 
                
                valid_mask = (lidar_uvs[:, 0] >= 0) & (lidar_uvs[:, 0] < 1920) & \
                             (lidar_uvs[:, 1] >= 0) & (lidar_uvs[:, 1] < 1280) & \
                             (lidar_points[:, 0] >= 2.0) & (lidar_points[:, 0] < 50.0)
                             
                if valid_mask.any():
                    v_uvs = lidar_uvs[valid_mask]
                    v_depths = lidar_points[valid_mask, 0]
                    
                    sort_idx = np.argsort(v_depths)[::-1]
                    v_uvs, v_depths = v_uvs[sort_idx], v_depths[sort_idx]
                    
                    u_feat = np.clip((v_uvs[:, 0] / 16.0).astype(np.int32), 0, 119)
                    v_feat = np.clip((v_uvs[:, 1] / 16.0).astype(np.int32), 0, 79)
                    d_bins = np.clip(((v_depths - 2.0) / 1.0).astype(np.int64), 0, 47)
                    depth_label[v_feat, u_feat] = d_bins

                # Note: Data augmentation (flips) is disabled here because flipping 
                # sequences requires complex ego-motion matrix inversions.
                step_data = {
                    'camera_image': img_tensor,       
                    'intrinsics': torch.from_numpy(intrinsics),          
                    'extrinsics': torch.from_numpy(extrinsics),
                    'depth_label': torch.from_numpy(depth_label),
                    'bboxes': torch.from_numpy(bboxes),
                    'num_valid_boxes': torch.tensor(valid_idx, dtype=torch.int32),
                    'lidar_points': torch.from_numpy(lidar_points), 
                    'lidar_uvs': torch.from_numpy(lidar_uvs)        
                }
            else:
                # Past frames (t_1, t_2) only need image and pose data for the caching module
                step_data = {
                    'camera_image': img_tensor,       
                    'intrinsics': torch.from_numpy(intrinsics),          
                    'extrinsics': torch.from_numpy(extrinsics),
                }
                
            sequence_data.append(step_data)
            
        return sequence_data


def temporal_collate_fn(batch):
    """
    Transforms a list of sequences into a dictionary of batched timesteps.
    Output structure: {'t_0': {...}, 't_1': {...}, 't_2': {...}}
    """
    seq_length = len(batch[0])
    collated = {}
    
    for step in range(seq_length):
        step_key = f't_{step}'
        step_batch = [item[step] for item in batch]
        
        collated[step_key] = {
            'camera_images': torch.stack([x['camera_image'] for x in step_batch]),
            'intrinsics': torch.stack([x['intrinsics'] for x in step_batch]),
            'extrinsics': torch.stack([x['extrinsics'] for x in step_batch])
        }
        
        # Only t_0 contains the ground truth targets
        if step == 0:
            collated[step_key]['depth_labels'] = torch.stack([x['depth_label'] for x in step_batch])
            collated[step_key]['bboxes'] = torch.stack([x['bboxes'] for x in step_batch])
            collated[step_key]['num_valid_boxes'] = torch.stack([x['num_valid_boxes'] for x in step_batch])
            
            # <--- ADD THIS BLOCK --->
            all_points = [x['lidar_points'] for x in step_batch]
            all_uvs = [x['lidar_uvs'] for x in step_batch]
            collated[step_key]['lidar_points'] = torch.cat(all_points, dim=0)
            collated[step_key]['lidar_uvs'] = torch.cat(all_uvs, dim=0)
            batch_indices = [torch.full((pts.shape[0],), i, dtype=torch.long) for i, pts in enumerate(all_points)]
            collated[step_key]['batch_indices'] = torch.cat(batch_indices, dim=0)
            
    return collated