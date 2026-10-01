#!/usr/bin/env python3
"""
Standalone 3D Detection Evaluation Harness for StudentBEVDetector.

Evaluates geometric detection accuracy (BEV AP and 3D AP at IoU 0.5 and 0.7)
across the Waymo Open Dataset (WOD) validation split.
Avoids any dependency on Google Waymo's official metrics_pb2 / TensorFlow evaluation
tools to eliminate environment, protobuf, and CUDA conflict issues.
"""

import os
import sys
import argparse
import glob
import json
import time
from pathlib import Path
from collections import defaultdict

# Inject project root into sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset

from src.perception.models.student_bev import StudentBEVDetector, se3_inverse
from src.perception.models.teacher_bev import WaymoBEVDetector
from src.perception.utils.dataset import (
    WaymoTemporalDataset,
    WaymoDataset,
    temporal_collate_fn,
    waymo_collate_fn
)
from src.perception.utils.nms_decoder import CenterNetDecoder


# Waymo Open Dataset Label Classes
CLASS_MAP = {
    1: 'Vehicle',
    2: 'Pedestrian',
    4: 'Cyclist'
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone 3D Detection Evaluator (BEV & 3D mAP @ 0.5 / 0.7)"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='student',
        choices=['student', 'teacher'],
        help='Model architecture to evaluate (student or teacher)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (.pt). Defaults to best_e2e_student_checkpoint.pt or best_waymo_bev_checkpoint.pt based on --model'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to dataset directory or specific .tfrecord file (defaults to data/raw/<split>/)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['val', 'train'],
        help='Dataset split to evaluate (default: val)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Compute device: "cuda", "cuda:0", "cpu" (default: auto-detect)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='DataLoader batch size (default: 2)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=2,
        help='Number of dataloader worker processes (default: 2)'
    )
    parser.add_argument(
        '--seq-length',
        type=int,
        default=3,
        help='Temporal sequence length for temporal student (default: 3)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to evaluate (default: all)'
    )
    parser.add_argument(
        '--score-thresh',
        type=float,
        default=0.15,
        help='Confidence score threshold for CenterNet decoding (default: 0.15)'
    )
    parser.add_argument(
        '--min-radius',
        type=float,
        default=1.5,
        help='NMS suppression radius in meters (default: 1.5)'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save quantitative metrics as JSON to scripts/benchmarks/eval_3d/results/'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=os.path.join(os.path.dirname(__file__), 'results'),
        help='Output directory for evaluation results'
    )
    return parser.parse_args()


# ==============================================================================
# IoU and Metric Calculation Utilities (Pure PyTorch / NumPy)
# ==============================================================================

def compute_bev_iou_matrix(boxes_a, boxes_b):
    """
    Computes pairwise 2D Bird's-Eye-View (BEV) IoU between two sets of boxes.
    boxes_a: [N, 4] -> [x, y, length, width]
    boxes_b: [M, 4] -> [x, y, length, width]
    Returns: [N, M] IoU matrix
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # Box A extents
    a_xmin = boxes_a[:, 0] - boxes_a[:, 2] / 2.0
    a_xmax = boxes_a[:, 0] + boxes_a[:, 2] / 2.0
    a_ymin = boxes_a[:, 1] - boxes_a[:, 3] / 2.0
    a_ymax = boxes_a[:, 1] + boxes_a[:, 3] / 2.0
    a_area = boxes_a[:, 2] * boxes_a[:, 3]

    # Box B extents
    b_xmin = boxes_b[:, 0] - boxes_b[:, 2] / 2.0
    b_xmax = boxes_b[:, 0] + boxes_b[:, 2] / 2.0
    b_ymin = boxes_b[:, 1] - boxes_b[:, 3] / 2.0
    b_ymax = boxes_b[:, 1] + boxes_b[:, 3] / 2.0
    b_area = boxes_b[:, 2] * boxes_b[:, 3]

    # Pairwise overlaps
    inter_xmin = np.maximum(a_xmin[:, None], b_xmin[None, :])
    inter_xmax = np.minimum(a_xmax[:, None], b_xmax[None, :])
    inter_ymin = np.maximum(a_ymin[:, None], b_ymin[None, :])
    inter_ymax = np.minimum(a_ymax[:, None], b_ymax[None, :])

    inter_w = np.clip(inter_xmax - inter_xmin, a_min=0.0, a_max=None)
    inter_h = np.clip(inter_ymax - inter_ymin, a_min=0.0, a_max=None)
    intersection = inter_w * inter_h

    union = a_area[:, None] + b_area[None, :] - intersection
    return intersection / (union + 1e-6)


def compute_3d_iou_matrix(boxes_a, boxes_b):
    """
    Computes pairwise 3D Axis-Aligned Volumetric IoU between two sets of boxes.
    boxes_a: [N, 6] -> [x, y, z, length, width, height]
    boxes_b: [M, 6] -> [x, y, z, length, width, height]
    Returns: [N, M] IoU matrix
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # Box A extents
    a_xmin = boxes_a[:, 0] - boxes_a[:, 3] / 2.0
    a_xmax = boxes_a[:, 0] + boxes_a[:, 3] / 2.0
    a_ymin = boxes_a[:, 1] - boxes_a[:, 4] / 2.0
    a_ymax = boxes_a[:, 1] + boxes_a[:, 4] / 2.0
    a_zmin = boxes_a[:, 2] - boxes_a[:, 5] / 2.0
    a_zmax = boxes_a[:, 2] + boxes_a[:, 5] / 2.0
    a_vol = boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]

    # Box B extents
    b_xmin = boxes_b[:, 0] - boxes_b[:, 3] / 2.0
    b_xmax = boxes_b[:, 0] + boxes_b[:, 3] / 2.0
    b_ymin = boxes_b[:, 1] - boxes_b[:, 4] / 2.0
    b_ymax = boxes_b[:, 1] + boxes_b[:, 4] / 2.0
    b_zmin = boxes_b[:, 2] - boxes_b[:, 5] / 2.0
    b_zmax = boxes_b[:, 2] + boxes_b[:, 5] / 2.0
    b_vol = boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]

    inter_xmin = np.maximum(a_xmin[:, None], b_xmin[None, :])
    inter_xmax = np.minimum(a_xmax[:, None], b_xmax[None, :])
    inter_ymin = np.maximum(a_ymin[:, None], b_ymin[None, :])
    inter_ymax = np.minimum(a_ymax[:, None], b_ymax[None, :])
    inter_zmin = np.maximum(a_zmin[:, None], b_zmin[None, :])
    inter_zmax = np.minimum(a_zmax[:, None], b_zmax[None, :])

    inter_l = np.clip(inter_xmax - inter_xmin, a_min=0.0, a_max=None)
    inter_w = np.clip(inter_ymax - inter_ymin, a_min=0.0, a_max=None)
    inter_h = np.clip(inter_zmax - inter_zmin, a_min=0.0, a_max=None)
    intersection = inter_l * inter_w * inter_h

    union = a_vol[:, None] + b_vol[None, :] - intersection
    return intersection / (union + 1e-6)

def compute_center_distance_matrix(boxes_a, boxes_b):
    """
    Computes pairwise 2D L2 Euclidean distance between the centers of two sets of boxes.
    boxes_a: [N, ...] where [:, 0] is X, [:, 1] is Y
    boxes_b: [M, ...] where [:, 0] is X, [:, 1] is Y
    Returns: [N, M] distance matrix
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.full((len(boxes_a), len(boxes_b)), fill_value=np.inf, dtype=np.float32)

    centers_a = boxes_a[:, :2] # [N, 2]
    centers_b = boxes_b[:, :2] # [M, 2]

    diff = centers_a[:, None, :] - centers_b[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1) # [N, M]
    return dist_matrix


def compute_ap(recalls, precisions):
    """
    Computes Average Precision (AP) using standard 11-point interpolation.
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        mask = recalls >= t
        if np.any(mask):
            p = np.max(precisions[mask])
        else:
            p = 0.0
        ap += p / 11.0
    return float(ap)


def evaluate_detection_ap(predictions_by_frame, ground_truths_by_frame, metric_thresh=0.5, mode='3d'):
    """
    Evaluates dataset-wide AP for a given collection of predictions and ground truths.

    predictions_by_frame: dict {frame_idx: list of dicts [{'x', 'y', 'z', 'length', 'width', 'height', 'score'}]}
    ground_truths_by_frame: dict {frame_idx: list of dicts [{'x', 'y', 'z', 'length', 'width', 'height'}]}
    metric_thresh: float (IoU threshold for '3d'/'bev', or Distance threshold in meters for 'center_dist')
    mode: '3d', 'bev', or 'center_dist'
    """
    # 1. Flatten all predictions with frame index
    all_preds = []
    total_gts = 0

    for f_idx, gt_boxes in ground_truths_by_frame.items():
        total_gts += len(gt_boxes)

    for f_idx, preds in predictions_by_frame.items():
        for p in preds:
            all_preds.append((f_idx, p['score'], p))

    if total_gts == 0:
        return 0.0, 0, 0

    if len(all_preds) == 0:
        return 0.0, 0, total_gts

    # 2. Sort predictions across all frames descending by confidence score
    all_preds.sort(key=lambda item: item[1], reverse=True)

    # Track matched ground truth boxes per frame
    matched_gt = {f_idx: set() for f_idx in ground_truths_by_frame.keys()}

    true_positives = np.zeros(len(all_preds), dtype=np.float32)
    false_positives = np.zeros(len(all_preds), dtype=np.float32)

    for i, (f_idx, score, pred_box) in enumerate(all_preds):
        gt_boxes = ground_truths_by_frame.get(f_idx, [])
        if len(gt_boxes) == 0:
            false_positives[i] = 1.0
            continue

        if mode == 'bev':
            pred_arr = np.array([[pred_box['x'], pred_box['y'], pred_box['length'], pred_box['width']]])
            gt_arr = np.array([[g['x'], g['y'], g['length'], g['width']] for g in gt_boxes])
            ious = compute_bev_iou_matrix(pred_arr, gt_arr)[0]
            best_gt_idx = int(np.argmax(ious))
            best_metric = ious[best_gt_idx]
            is_match = (best_metric >= metric_thresh)
        elif mode == '3d':
            pred_arr = np.array([[pred_box['x'], pred_box['y'], pred_box['z'],
                                  pred_box['length'], pred_box['width'], pred_box['height']]])
            gt_arr = np.array([[g['x'], g['y'], g['z'], g['length'], g['width'], g['height']] for g in gt_boxes])
            ious = compute_3d_iou_matrix(pred_arr, gt_arr)[0]
            best_gt_idx = int(np.argmax(ious))
            best_metric = ious[best_gt_idx]
            is_match = (best_metric >= metric_thresh)
        else: # center_dist
            pred_arr = np.array([[pred_box['x'], pred_box['y']]])
            gt_arr = np.array([[g['x'], g['y']] for g in gt_boxes])
            dists = compute_center_distance_matrix(pred_arr, gt_arr)[0]
            best_gt_idx = int(np.argmin(dists))
            best_metric = dists[best_gt_idx]
            is_match = (best_metric <= metric_thresh)

        if is_match and best_gt_idx not in matched_gt[f_idx]:
            true_positives[i] = 1.0
            matched_gt[f_idx].add(best_gt_idx)
        else:
            false_positives[i] = 1.0

    cum_tp = np.cumsum(true_positives)
    cum_fp = np.cumsum(false_positives)

    recalls = cum_tp / (total_gts + 1e-6)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-6)

    ap = compute_ap(recalls, precisions)
    return ap, int(np.sum(true_positives)), total_gts


def classify_box_dimension(length, width, height):
    """
    Dimension-based class prior classifier for CenterPoint / CenterNet outputs:
    1: Vehicle (Car / Truck / Bus)
    2: Pedestrian
    4: Cyclist
    """
    if length < 1.4 and width < 1.4:
        return 2  # Pedestrian
    elif length < 2.6 and width < 1.4:
        return 4  # Cyclist
    else:
        return 1  # Vehicle


# ==============================================================================
# Main Evaluation Harness
# ==============================================================================

def main():
    args = parse_args()

    # 1. Resolve Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("   STANDALONE 3D DETECTION EVALUATION HARNESS (STUDENT BEV)")
    print("=" * 80)
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() and 'cuda' in str(device) else 'CPU'})")

    # 2. Resolve Checkpoint Path
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = 'best_waymo_bev_checkpoint.pt' if args.model == 'teacher' else 'best_e2e_student_checkpoint.pt'
        
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    print(f"Checkpoint    : {ckpt_path}")

    # 3. Inspect Checkpoint & Architecture Mode
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'ema_state_dict' in checkpoint:
        state_dict = checkpoint['ema_state_dict']
        print(f"Weights Source: EMA weights (Epoch {checkpoint.get('epoch', 'N/A')}, Val Score: {checkpoint.get('val_score', 'N/A')})")
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Weights Source: Model state_dict (Epoch {checkpoint.get('epoch', 'N/A')})")
    else:
        state_dict = checkpoint
        print("Weights Source: Raw state dictionary")

    # Strip potential 'module.' prefixes
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # 4. Instantiate Model
    if args.model == 'teacher':
        print("Architecture  : WaymoBEVDetector (Teacher)")
        has_temporal = False
        model = WaymoBEVDetector().to(device)
    else:
        has_temporal = any('temporal_fusion' in k for k in clean_state_dict.keys())
        print(f"Architecture  : StudentBEVDetector (use_temporal={has_temporal})")
        model = StudentBEVDetector(bev_h=160, bev_w=160, feature_channels=64, use_temporal=has_temporal).to(device)
        
    load_res = model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    if load_res.missing_keys:
        print(f"Note: {len(load_res.missing_keys)} missing keys during loading (non-critical).")

    # 5. Resolve Dataset Files
    if args.data_path:
        target_path = Path(args.data_path)
        if target_path.is_file():
            tfrecord_files = [str(target_path)]
        elif target_path.is_dir():
            tfrecord_files = sorted(glob.glob(str(target_path / '*.tfrecord')))
        else:
            tfrecord_files = sorted(glob.glob(args.data_path))
    else:
        split_dir = os.path.join(PROJECT_ROOT, 'data', 'raw', args.split)
        tfrecord_files = sorted(glob.glob(os.path.join(split_dir, '*.tfrecord')))

    if not tfrecord_files:
        raise FileNotFoundError(
            f"No .tfrecord files found for evaluation. "
            f"Check '--data-path' or verify that data exists in 'data/raw/{args.split}/'."
        )

    print(f"Dataset Split : {args.split} ({len(tfrecord_files)} TFRecord files discovered)")
    for f in tfrecord_files[:3]:
        print(f"  -> {os.path.basename(f)}")
    if len(tfrecord_files) > 3:
        print(f"  ... and {len(tfrecord_files) - 3} more files")

    # 6. Build Dataloader
    if args.model == 'student' and has_temporal:
        dataset_list = [
            WaymoTemporalDataset(f, is_train=False, seq_length=args.seq_length)
            for f in tfrecord_files
        ]
        collate_fn = temporal_collate_fn
    else:
        dataset_list = [
            WaymoDataset(f, is_train=False, num_sweeps=1)
            for f in tfrecord_files
        ]
        collate_fn = waymo_collate_fn

    eval_dataset = ConcatDataset(dataset_list)
    total_frames = len(eval_dataset)
    print(f"Total Frames  : {total_frames} available in split")

    dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if 'cuda' in str(device) else False
    )

    # 7. Initialize CenterNet Decoder
    decoder = CenterNetDecoder(
        x_range=(0.0, 70.0),
        y_range=(-40.0, 40.0),
        bev_h=160,
        bev_w=160,
        threshold=args.score_thresh,
        min_radius=args.min_radius
    )

    # 8. Accumulation Structures for Predictions & Ground Truths
    # Format: dict[class_id] -> dict[frame_idx] -> list of boxes
    # Class ID 0 represents overall (class-agnostic across Vehicles, Pedestrians, Cyclists)
    preds_per_class = {c: defaultdict(list) for c in [0, 1, 2, 4]}
    gts_per_class = {c: defaultdict(list) for c in [0, 1, 2, 4]}

    print("\nStarting evaluation inference loop...")
    eval_start_time = time.time()
    evaluated_frames = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if args.max_frames and evaluated_frames >= args.max_frames:
                break

            current_bs = 0

            if args.model == 'teacher':
                cam = batch['camera_images'].to(device, non_blocking=True)
                lidar_points = batch['lidar_points'].to(device, non_blocking=True)
                batch_indices = batch['batch_indices'].to(device, non_blocking=True)
                lidar_uvs = batch['lidar_uvs'].to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if 'cuda' in str(device) else torch.bfloat16):
                    s_out = model(lidar_points, batch_indices, cam, lidar_uvs)

                pred_dict = {
                    'bev_occupancy': s_out['bev_occupancy'].float(),
                    'dimensions': s_out['dimensions'].float(),
                    'orientation': s_out['orientation'].float(),
                    'offset': s_out['offset'].float()
                }
                current_bs = cam.shape[0]
                gt_bboxes = batch['bboxes']
                gt_valid = batch['num_valid_boxes']
                
            elif args.model == 'student' and has_temporal:
                # Sequence forward pass (t_2 -> t_1 -> t_0)
                past_features = None
                past_extrinsics = None

                for step in reversed(range(args.seq_length)):
                    step_key = f't_{step}'
                    step_data = batch[step_key]

                    cam = step_data['camera_images'].to(device, non_blocking=True)
                    intrin = step_data['intrinsics'].to(device, non_blocking=True)
                    extrin = step_data['extrinsics'].to(device, non_blocking=True)
                    extrin_inv = se3_inverse(extrin)

                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if 'cuda' in str(device) else torch.bfloat16):
                        s_out = model(
                            cam, intrin, extrin_inv,
                            past_features=past_features,
                            current_extrinsics=extrin,
                            past_extrinsics=past_extrinsics
                        )

                    past_features = s_out['bev_features']
                    past_extrinsics = extrin

                    if step == 0:
                        pred_dict = {
                            'bev_occupancy': s_out['bev_occupancy'].float(),
                            'dimensions': s_out['dimensions'].float(),
                            'orientation': s_out['orientation'].float(),
                            'offset': s_out['offset'].float()
                        }
                        current_bs = cam.shape[0]
                        gt_bboxes = step_data['bboxes']
                        gt_valid = step_data['num_valid_boxes']

            else:
                # Single-frame pass
                cam = batch['camera_images'].to(device, non_blocking=True)
                intrin = batch['intrinsics'].to(device, non_blocking=True)
                extrin = batch['extrinsics'].to(device, non_blocking=True)
                extrin_inv = se3_inverse(extrin)

                with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if 'cuda' in str(device) else torch.bfloat16):
                    s_out = model(cam, intrin, extrin_inv)

                pred_dict = {
                    'bev_occupancy': s_out['bev_occupancy'].float(),
                    'dimensions': s_out['dimensions'].float(),
                    'orientation': s_out['orientation'].float(),
                    'offset': s_out['offset'].float()
                }
                current_bs = cam.shape[0]
                gt_bboxes = batch['bboxes']
                gt_valid = batch['num_valid_boxes']

            # Decode Bounding Boxes for batch
            decoded_batch = decoder.decode(pred_dict)

            # Process Ground Truths and Predictions for each frame in batch
            for b_idx in range(current_bs):
                global_frame_idx = evaluated_frames + b_idx
                if args.max_frames and global_frame_idx >= args.max_frames:
                    break

                # 1. Ground Truth boxes
                num_v = int(gt_valid[b_idx].item())
                frame_gt_boxes = gt_bboxes[b_idx, :num_v].cpu().numpy()

                for gt_row in frame_gt_boxes:
                    cls_id = int(gt_row[0])
                    # Only collect valid foreground classes
                    if cls_id in CLASS_MAP:
                        gt_obj = {
                            'x': float(gt_row[3]),
                            'y': float(gt_row[4]),
                            'z': float(gt_row[5]),
                            'length': float(gt_row[6]),
                            'width': float(gt_row[7]),
                            'height': float(gt_row[8]),
                            'heading': float(gt_row[9]),
                            'class_id': cls_id
                        }
                        # Add to class-specific GT list
                        gts_per_class[cls_id][global_frame_idx].append(gt_obj)
                        # Add to overall GT list (0)
                        gts_per_class[0][global_frame_idx].append(gt_obj)

                # 2. Predicted boxes
                preds = decoded_batch[b_idx]
                for p in preds:
                    pred_cls = classify_box_dimension(p['length'], p['width'], p['height'])
                    p_obj = dict(p)
                    p_obj['class_id'] = pred_cls

                    # Add to assigned class
                    preds_per_class[pred_cls][global_frame_idx].append(p_obj)
                    # Add to overall predictions (0)
                    preds_per_class[0][global_frame_idx].append(p_obj)

            evaluated_frames += current_bs

            if (batch_idx + 1) % 10 == 0 or (args.max_frames and evaluated_frames >= args.max_frames):
                elapsed = time.time() - eval_start_time
                fps = evaluated_frames / (elapsed + 1e-6)
                target_total = args.max_frames if args.max_frames else total_frames
                print(f"  [Progress] Evaluated {evaluated_frames:04d}/{target_total} frames | Speed: {fps:.1f} FPS | Elapsed: {elapsed:.1f}s")

    total_eval_time = time.time() - eval_start_time
    print(f"\nInference complete in {total_eval_time:.2f}s ({evaluated_frames / total_eval_time:.1f} FPS). Computing AP metrics...")

    # 9. Compute AP and mAP Metrics
    metrics_summary = {}

    eval_classes = [
        (1, 'Vehicle (Car)'),
        (2, 'Pedestrian'),
        (4, 'Cyclist'),
        (0, 'Overall (All Classes)')
    ]

    for c_id, c_name in eval_classes:
        preds = preds_per_class[c_id]
        gts = gts_per_class[c_id]

        bev_ap_50, _, total_gt = evaluate_detection_ap(preds, gts, metric_thresh=0.5, mode='bev')
        bev_ap_70, _, _ = evaluate_detection_ap(preds, gts, metric_thresh=0.7, mode='bev')

        iou_3d_50, _, _ = evaluate_detection_ap(preds, gts, metric_thresh=0.5, mode='3d')
        iou_3d_70, _, _ = evaluate_detection_ap(preds, gts, metric_thresh=0.7, mode='3d')

        cdist_05, _, _ = evaluate_detection_ap(preds, gts, metric_thresh=0.5, mode='center_dist')
        cdist_10, _, _ = evaluate_detection_ap(preds, gts, metric_thresh=1.0, mode='center_dist')
        cdist_20, _, _ = evaluate_detection_ap(preds, gts, metric_thresh=2.0, mode='center_dist')
        cdist_40, _, _ = evaluate_detection_ap(preds, gts, metric_thresh=4.0, mode='center_dist')

        metrics_summary[c_name] = {
            'class_id': c_id,
            'gt_count': total_gt,
            'bev_ap_0.5': bev_ap_50,
            'bev_ap_0.7': bev_ap_70,
            '3d_ap_0.5': iou_3d_50,
            '3d_ap_0.7': iou_3d_70,
            'cdist_ap_0.5m': cdist_05,
            'cdist_ap_1.0m': cdist_10,
            'cdist_ap_2.0m': cdist_20,
            'cdist_ap_4.0m': cdist_40
        }

    # 10. Display Summary Table
    print("\n" + "=" * 128)
    print("                                      3D OBJECT DETECTION EVALUATION RESULTS (WITH NDS METRICS)")
    print("=" * 128)
    print(f"{'Class':<22} | {'GT Count':<9} | {'BEV AP@0.5':<10} | {'3D AP@0.5':<10} | {'CD AP@0.5m':<10} | {'CD AP@1.0m':<10} | {'CD AP@2.0m':<10} | {'CD AP@4.0m':<10}")
    print("-" * 128)

    for c_id, c_name in eval_classes:
        m = metrics_summary[c_name]
        is_overall = (c_id == 0)
        if is_overall:
            print("-" * 128)
        print(
            f"{c_name:<22} | {m['gt_count']:<9d} | {m['bev_ap_0.5']:<10.4f} | "
            f"{m['3d_ap_0.5']:<10.4f} | {m['cdist_ap_0.5m']:<10.4f} | "
            f"{m['cdist_ap_1.0m']:<10.4f} | {m['cdist_ap_2.0m']:<10.4f} | {m['cdist_ap_4.0m']:<10.4f}"
        )
    print("=" * 128 + "\n")

    # 11. Optionally Save Results to JSON
    if args.save_results:
        os.makedirs(args.results_dir, exist_ok=True)
        results_file = os.path.join(args.results_dir, 'metrics.json')

        payload = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'checkpoint': os.path.abspath(ckpt_path),
            'split': args.split,
            'frames_evaluated': evaluated_frames,
            'total_time_seconds': round(total_eval_time, 2),
            'fps': round(evaluated_frames / total_eval_time, 2),
            'score_threshold': args.score_thresh,
            'min_radius': args.min_radius,
            'metrics': metrics_summary
        }

        with open(results_file, 'w') as f:
            json.dump(payload, f, indent=4)

        print(f"[Results Saved] Quantitative metrics successfully written to:\n  -> {results_file}\n")


if __name__ == '__main__':
    main()
