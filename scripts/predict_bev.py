import os
import sys
import glob
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset

# Inject root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.perception.models.teacher_bev import WaymoBEVDetector
from src.perception.utils.dataset import WaymoDataset, waymo_collate_fn
from src.perception.utils.target_encoder import BEVGridEncoder

def visualize_teacher(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing LiDAR Teacher Visualizer on {device}...")

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
                occ_logits = predictions['bev_occupancy']
                # Horizontal coordinate synchronization
                occ_logits = torch.flip(occ_logits, dims=[-1])
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
            
            axes[0].imshow(image_rgb)
            axes[0].set_title('Front Camera', fontsize=14, color='white')
            axes[0].axis('off')
            
            axes[1].imshow(gt_occ, cmap='Blues', vmin=0, vmax=1)
            axes[1].set_title('Ground Truth (LiDAR Boxes)', fontsize=14, color='white')
            axes[1].axis('off')
            
            axes[2].imshow(scaled_pred, cmap='magma', vmin=0, vmax=1.0)
            axes[2].set_title(f'Teacher Prediction (Auto-Scaled | Peak: {p_max:.3f})', fontsize=14, color='white')
            axes[2].axis('off')

            plt.tight_layout()
            out_path = os.path.join(args.out_dir, f'teacher_eval_{count:03d}.png')
            plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close()
            
            print(f"Saved visualization: {out_path} (Peak: {p_max:.4f}, GT Boxes: {valid_boxes})")
            count += 1
            if count >= args.num_samples:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Multi-Modal Teacher')
    parser.add_argument('--checkpoint', type=str, default='best_waymo_bev_checkpoint.pt')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--out_dir', type=str, default='runs/teacher_visualizations')
    args = parser.parse_args()
    
    visualize_teacher(args)