import torch
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, ConcatDataset

from model import WaymoBEVDetector 
from utils.dataset import WaymoDataset
from utils.target_encoder import BEVGridEncoder
from utils.validate import validate_model
from loss import BEVFocalLoss, CombinedBEVLoss

def save_bev_side_by_side(pred_prob, target_grid, output_path='prediction_bev.jpg', threshold=0.0):
    """
    Renders a side-by-side comparison using TensorBoard-style min-max auto-scaling 
    so low-magnitude probability patterns become clearly visible.
    """
    pred_np = np.squeeze(pred_prob.detach().cpu().numpy())
    target_np = np.squeeze(target_grid.detach().cpu().numpy())

    # --- TENSORBOARD-STYLE AUTO-SCALING ---
    p_min, p_max = pred_np.min(), pred_np.max()
    if p_max > p_min:
        scaled_pred = (pred_np - p_min) / (p_max - p_min)
    else:
        scaled_pred = np.zeros_like(pred_np)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='#1E1E1E')
    fig.suptitle("WaymoBEVDetector: Model Prediction vs Ground Truth", color='white', fontsize=14, fontweight='bold')

    # Panel 1: Auto-scaled Model Prediction Heatmap
    im0 = axes[0].imshow(scaled_pred, cmap='magma', vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Model Prediction (Auto-Scaled | Raw Peak: {p_max:.4f})", color='white', fontsize=11)
    axes[0].axis('off')
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar0.ax, 'yticklabels'), color='white')

    # Panel 2: Ground Truth BEV Grid
    neon_cmap = ListedColormap(['black', '#00FFCC'])
    axes[1].imshow(target_np, cmap=neon_cmap, interpolation='nearest')
    axes[1].set_title("Ground Truth BEV Raster", color='white', fontsize=11)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved side-by-side BEV comparison map to {output_path}")

def custom_collate_fn(batch):
    return {
        'timestamp': torch.stack([item['timestamp'] for item in batch]),
        'front_image': torch.stack([item['front_image'] for item in batch]),
        'bboxes': torch.stack([item['bboxes'] for item in batch]),
        'num_valid_boxes': torch.stack([item['num_valid_boxes'] for item in batch]),
        'lidar_points': [item['lidar_points'] for item in batch]
    }

def run_validation_prediction(checkpoint_path='best_waymo_bev_checkpoint.pt', output_path='prediction_bev.jpg', bev_threshold=0.004, frame_index=None):
    print(f"Loading model from {checkpoint_path}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = WaymoBEVDetector() 
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # --- RESTORE EMA WEIGHTS & BATCHNORM STATS ---
    base_state = checkpoint.get('model_state_dict', {})
    ema_state = checkpoint.get('ema_model_state_dict', {})
    
    if ema_state:
        print("Injecting healthy BatchNorm running statistics from base model into EMA...")
        for key, value in base_state.items():
            if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                ema_key = 'module.' + key 
                if ema_key in ema_state:
                    ema_state[ema_key] = value
        state_dict = ema_state
    else:
        state_dict = base_state if base_state else checkpoint

    clean_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith('module.') else k
        clean_state_dict[new_key] = v
        
    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    print(f"Checkpoint Loaded | Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    
    # --- SETUP VALIDATION DATASET ---
    val_files = glob.glob('data/raw/val/*.tfrecord')
    if not val_files:
        raise FileNotFoundError("No .tfrecord files found in data/raw/val/")

    val_datasets = [WaymoDataset(tfrecord_path=f) for f in val_files]
    val_dataset = ConcatDataset(val_datasets)
    print(f"Total validation frames available: {len(val_dataset)}")
    
    encoder = BEVGridEncoder(x_range=(0.0, 80.0), y_range=(-40.0, 40.0), resolution=0.5)

    # --- EXTRACT SPECIFIC FRAME OR SCAN FOR CENTER TRAFFIC ---
    if frame_index is not None:
        if frame_index >= len(val_dataset):
            print(f"Error: Requested frame_index {frame_index} exceeds dataset size ({len(val_dataset)}).")
            return
        sample = val_dataset[frame_index]
        target_batch = {
            'front_image': sample['front_image'].unsqueeze(0),
            'bboxes': sample['bboxes'].unsqueeze(0),
            'num_valid_boxes': sample['num_valid_boxes'].unsqueeze(0)
        }
        print(f"Loaded requested Frame {frame_index} containing {sample['num_valid_boxes'].item()} ground truth vehicles.")
    else:
        print(f"Scanning validation dataset for a 'center-lane traffic' frame...")
        target_batch = None
        val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True, collate_fn=custom_collate_fn)
        
        for batch in val_dataloader:
            for b_idx in range(batch['front_image'].shape[0]):
                valid_boxes = batch['num_valid_boxes'][b_idx].item()
                bboxes = batch['bboxes'][b_idx]
                
                # Count how many cars are directly in front of the ego vehicle (y between -1.5 and 1.5 meters)
                center_cars = 0
                for i in range(valid_boxes):
                    y = bboxes[i, 4].item()
                    x = bboxes[i, 3].item()
                    # Filter for cars straight ahead in the ego-lane
                    if abs(y) < 1.5 and x > 5.0:
                        center_cars += 1
                        
                if center_cars >= 2: # Frame with at least 2 cars directly ahead
                    target_batch = {
                        'front_image': batch['front_image'][b_idx:b_idx+1],
                        'bboxes': batch['bboxes'][b_idx:b_idx+1],
                        'num_valid_boxes': batch['num_valid_boxes'][b_idx:b_idx+1]
                    }
                    print(f"Found ideal frame! Contains {valid_boxes} total vehicles, with {center_cars} directly in the center lane.")
                    break
            if target_batch is not None:
                break
                
        if target_batch is None:
            print("Could not find a heavy center-lane frame. Defaulting to frame 0.")
            sample = val_dataset[0]
            target_batch = {
                'front_image': sample['front_image'].unsqueeze(0),
                'bboxes': sample['bboxes'].unsqueeze(0),
                'num_valid_boxes': sample['num_valid_boxes'].unsqueeze(0)
            }

    # Save debug camera view
    rgb_np = target_batch['front_image'][0, :3].permute(1, 2, 0).numpy().astype(np.uint8)
    cv2.imwrite('debug_input_frame.jpg', cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
    print("Saved 'debug_input_frame.jpg' for visual verification.")

    # Run forward pass for visualization
    images = target_batch['front_image'].to(device, dtype=torch.float32)
    targets_dict = encoder.encode(target_batch['bboxes'], target_batch['num_valid_boxes'])
    target_grid = targets_dict['bev_occupancy']

    print("Running inference visualization...")
    with torch.no_grad(), torch.amp.autocast('cuda'):
        predictions = model(images)
        pred_prob = torch.sigmoid(predictions['bev_occupancy'])
        
        # --- HORIZONTAL FLIP FIX (Optional) ---
        # Uncomment the line below to align the lateral coordinate systems 
        # for vehicles that are outside the center lane.
        # pred_prob = torch.flip(pred_prob, dims=[-1])
        
    print(f"Peak prediction probability: {pred_prob.max().item():.4f}")
    
    save_bev_side_by_side(pred_prob, target_grid, output_path=output_path, threshold=bev_threshold)

if __name__ == '__main__':
    checkpoint_file = 'best_waymo_bev_checkpoint.pt'
    # Set frame_index=None to trigger the center-lane traffic scanner
    run_validation_prediction(checkpoint_path=checkpoint_file, output_path='prediction_output.jpg', frame_index=5)