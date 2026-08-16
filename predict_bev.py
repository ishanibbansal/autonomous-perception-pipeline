import torch
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, ConcatDataset

from src.perception.models.teacher_bev import WaymoBEVDetector
from src.perception.utils.dataset import WaymoDataset, waymo_collate_fn
from src.perception.utils.target_encoder import BEVGridEncoder

def save_bev_side_by_side(pred_prob, target_grid, output_path='prediction_bev.jpg'):
    """
    Renders a side-by-side comparison using TensorBoard-style min-max auto-scaling 
    so low-magnitude probability patterns become clearly visible.
    """
    pred_np = np.squeeze(pred_prob.detach().cpu().numpy())
    target_np = np.squeeze(target_grid.detach().cpu().numpy())

    p_min, p_max = pred_np.min(), pred_np.max()
    if p_max > p_min:
        scaled_pred = (pred_np - p_min) / (p_max - p_min)
    else:
        scaled_pred = np.zeros_like(pred_np)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='#1E1E1E')
    fig.suptitle("Teacher Model: Prediction vs Ground Truth", color='white', fontsize=14, fontweight='bold')

    im0 = axes[0].imshow(scaled_pred, cmap='magma', vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Model Prediction (Auto-Scaled | Raw Peak: {p_max:.4f})", color='white', fontsize=11)
    axes[0].axis('off')
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar0.ax, 'yticklabels'), color='white')

    neon_cmap = ListedColormap(['black', '#00FFCC'])
    axes[1].imshow(target_np, cmap=neon_cmap, interpolation='nearest')
    axes[1].set_title("Ground Truth BEV Raster", color='white', fontsize=11)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved side-by-side BEV comparison map to {output_path}")

def run_validation_prediction(checkpoint_path='best_waymo_bev_checkpoint.pt', output_path='prediction_output.jpg', frame_index=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {checkpoint_path} onto {device}...")
    
    model = WaymoBEVDetector().to(device)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Successfully loaded Teacher model from Epoch {checkpoint.get('epoch', 'N/A')}")

    model.eval()
    
    val_files = glob.glob('data/raw/val/*.tfrecord')
    if not val_files:
        raise FileNotFoundError("No .tfrecord files found in data/raw/val/")

    val_datasets = [WaymoDataset(tfrecord_path=f, is_train=False, num_sweeps=3) for f in val_files]
    val_dataset = ConcatDataset(val_datasets)
    print(f"Total validation frames available: {len(val_dataset)}")
    
    encoder = BEVGridEncoder(x_range=(0.0, 70.0), y_range=(-40.0, 40.0), bev_h=160, bev_w=160)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=waymo_collate_fn)

    target_batch = None
    if frame_index is not None:
        print(f"Fast-forwarding to requested Frame {frame_index}...")
        for i, batch in enumerate(val_dataloader):
            if i == frame_index:
                target_batch = batch
                break
    else:
        print(f"Scanning validation dataset for a 'center-lane traffic' frame...")
        for batch in val_dataloader:
            valid_boxes = batch['num_valid_boxes'][0].item()
            bboxes = batch['bboxes'][0]
            
            center_cars = 0
            for i in range(valid_boxes):
                y = bboxes[i, 4].item()
                x = bboxes[i, 3].item()
                if abs(y) < 1.5 and x > 5.0:
                    center_cars += 1
                    
            if center_cars >= 2:
                target_batch = batch
                print(f"Found ideal frame! Contains {valid_boxes} total vehicles, with {center_cars} directly in the center lane.")
                break
                
        if target_batch is None:
            print("Could not find a heavy center-lane frame. Defaulting to first frame.")
            target_batch = next(iter(val_dataloader))

    rgb_np = target_batch['camera_images'][0].permute(1, 2, 0).numpy()
    rgb_np = (rgb_np * 255.0).astype(np.uint8)
    cv2.imwrite('debug_input_frame.jpg', cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
    print("Saved 'debug_input_frame.jpg' for visual verification.")

    print("Running Teacher Model forward pass...")
    
    lidar_points = target_batch['lidar_points'].to(device)
    batch_indices = target_batch['batch_indices'].to(device)
    camera_images = target_batch['camera_images'].to(device)
    lidar_uvs = target_batch['lidar_uvs'].to(device)
    
    targets_dict = encoder.encode(target_batch['bboxes'], target_batch['num_valid_boxes'])
    target_grid = targets_dict['bev_occupancy']

    with torch.no_grad(), torch.amp.autocast('cuda'):
        predictions = model(lidar_points, batch_indices, camera_images, lidar_uvs)
        pred_prob = torch.sigmoid(predictions['bev_occupancy'])
        
        # Apply the required horizontal mirror to evaluate properly aligned visualizations
        pred_prob = torch.flip(pred_prob, dims=[-1])
        
    print(f"Peak prediction probability: {pred_prob.max().item():.4f}")
    save_bev_side_by_side(pred_prob, target_grid, output_path=output_path)

if __name__ == '__main__':
    run_validation_prediction(
        checkpoint_path='best_waymo_bev_checkpoint.pt', 
        output_path='prediction_output.jpg', 
        frame_index=None
    )