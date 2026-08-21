import os
import sys
import glob
import math
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader, ConcatDataset

# Inject root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.perception.models.teacher_bev import WaymoBEVDetector
from src.perception.utils.dataset import WaymoDataset, waymo_collate_fn
from src.perception.utils.target_encoder import BEVGridEncoder
from src.perception.utils.nms_decoder import CenterNetDecoder

def get_box_polygon_grid(x, y, length, width, heading, res_x, res_y, x_min=0.0, y_min=-40.0, bev_h=160, bev_w=160):
    """
    Computes 4 bounding box corner coordinates projected into the 160x160 image grid.
    """
    # 4 corners in local object frame
    dx = np.array([-length / 2, length / 2, length / 2, -length / 2])
    dy = np.array([-width / 2, -width / 2, width / 2, width / 2])

    # Rotate by heading
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    x_rot = x + dx * cos_h - dy * sin_h
    y_rot = y + dx * sin_h + dy * cos_h

    # Convert physical (X, Y) to grid indices (col, row)
    cols = (bev_w - 1) - ((y_rot - y_min) / res_y)
    rows = (bev_h - 1) - ((x_rot - x_min) / res_x)

    return np.stack([cols, rows], axis=-1)

def visualize_teacher(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing LiDAR Teacher + NMS Visualizer on {device}...")

    val_files = glob.glob('data/raw/val/*.tfrecord')
    if not val_files:
        print("Error: No validation TFRecords found.")
        return

    val_datasets = [WaymoDataset(f, is_train=False, num_sweeps=3) for f in val_files]
    dataset = ConcatDataset(val_datasets)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=args.shuffle, collate_fn=waymo_collate_fn)

    model = WaymoBEVDetector().to(device)
    print(f"Loading weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    encoder = BEVGridEncoder(x_range=(0.0, 70.0), y_range=(-40.0, 40.0), bev_h=160, bev_w=160)
    decoder = CenterNetDecoder(x_range=(0.0, 70.0), y_range=(-40.0, 40.0), bev_h=160, bev_w=160, threshold=args.threshold)
    
    os.makedirs(args.out_dir, exist_ok=True)

    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            valid_boxes = batch['num_valid_boxes'][0].item()
            if valid_boxes == 0 and not args.shuffle:
                continue

            lidar_points = batch['lidar_points'].to(device)
            batch_indices = batch['batch_indices'].to(device)
            camera_images = batch['camera_images'].to(device)
            lidar_uvs = batch['lidar_uvs'].to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                predictions = model(lidar_points, batch_indices, camera_images, lidar_uvs)
                
                # Extract physical 3D boxes via NMS decoder
                decoded_boxes = decoder.decode(predictions)[0]

                occ_logits = torch.flip(predictions['bev_occupancy'], dims=[-1])
                occ_probs = torch.sigmoid(occ_logits)[0, 0].cpu().numpy()

            encoded_targets = encoder.encode(batch['bboxes'], batch['num_valid_boxes'])
            gt_occ = encoded_targets['bev_occupancy'][0, 0].cpu().numpy()

            p_min, p_max = occ_probs.min(), occ_probs.max()
            if p_max > p_min:
                scaled_pred = (occ_probs - p_min) / (p_max - p_min)
            else:
                scaled_pred = np.zeros_like(occ_probs)

            image_tensor = camera_images[0].cpu().numpy()
            image_rgb = np.clip(image_tensor.transpose(1, 2, 0), 0, 1)

            fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor='#121212')
            
            # Panel 1: Front Camera
            axes[0].imshow(image_rgb)
            axes[0].set_title('Front Camera', fontsize=14, color='white')
            axes[0].axis('off')
            
            # Panel 2: Ground Truth Raster + GT Box Outlines
            axes[1].imshow(gt_occ, cmap='Blues', vmin=0, vmax=1)
            axes[1].set_title(f'Ground Truth ({valid_boxes} Boxes)', fontsize=14, color='white')
            axes[1].axis('off')
            
            gt_raw = batch['bboxes'][0].cpu().numpy()
            for i in range(valid_boxes):
                gx, gy = gt_raw[i, 3], gt_raw[i, 4]
                gl, gw = gt_raw[i, 6], gt_raw[i, 7]
                gh = gt_raw[i, 9]
                poly = get_box_polygon_grid(gx, gy, gl, gw, gh, encoder.res_x, encoder.res_y)
                polygon = patches.Polygon(poly, linewidth=1.5, edgecolor='#00FFCC', facecolor='none')
                axes[1].add_patch(polygon)
            
            # Panel 3: Auto-Scaled Heatmap + NMS Extracted Bounding Boxes
            axes[2].imshow(scaled_pred, cmap='magma', vmin=0, vmax=1.0)
            axes[2].set_title(f'NMS Decoded ({len(decoded_boxes)} Detected | Peak: {p_max:.3f})', fontsize=14, color='white')
            axes[2].axis('off')

            for box in decoded_boxes:
                poly = get_box_polygon_grid(
                    box['x'], box['y'], box['length'], box['width'], box['heading'],
                    decoder.res_x, decoder.res_y
                )
                polygon = patches.Polygon(poly, linewidth=1.5, edgecolor='#00FF66', facecolor='none')
                axes[2].add_patch(polygon)
                
                # Mark centroid dot
                col = (160 - 1) - ((box['y'] - (-40.0)) / decoder.res_y)
                row = (160 - 1) - ((box['x'] - 0.0) / decoder.res_x)
                axes[2].plot(col, row, 'ro', markersize=3)

            plt.tight_layout()
            out_path = os.path.join(args.out_dir, f'nms_eval_{count:03d}.png')
            plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close()
            
            print(f"Saved: {out_path} -> GT: {valid_boxes} boxes | Detected: {len(decoded_boxes)} boxes")
            count += 1
            if count >= args.num_samples:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Multi-Modal Teacher with NMS Decoder')
    parser.add_argument('--checkpoint', type=str, default='best_waymo_bev_checkpoint.pt')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.25, help='NMS peak confidence threshold')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--out_dir', type=str, default='runs/teacher_nms_visualizations')
    args = parser.parse_args()
    
    visualize_teacher(args)