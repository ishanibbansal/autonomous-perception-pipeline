import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.perception.models.student_bev import StudentBEVDetector
from src.perception.utils.dataset import WaymoDataset
from src.perception.utils.target_encoder import BEVGridEncoder

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Inference on: {device}")

    # 1. Load the Best Student Model
    checkpoint_path = 'best_student_bev_checkpoint.pt'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Cannot find checkpoint: {checkpoint_path}")

    student = StudentBEVDetector().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    student.load_state_dict(checkpoint['model_state_dict'])
    student.eval()
    print(f"Loaded student model from Epoch {checkpoint.get('epoch', 'Unknown')}")

    # 2. Load a Single Validation Frame
    val_files = glob.glob('data/raw/val/*.tfrecord')
    if not val_files:
        raise FileNotFoundError("No validation TFRecords found in data/raw/val/")
    
    # Initialize dataset (turn off sweeps since Student only needs the current image)
    dataset = WaymoDataset(tfrecord_path=val_files[0], is_train=False, num_sweeps=1)
    
    # Pick a random frame that actually has boxes
    frame_idx = random.randint(0, len(dataset) - 1)
    sample = dataset[frame_idx]
    
    # 3. Prepare Tensors (Add Batch Dimension)
    camera_image = sample['camera_image'].unsqueeze(0).to(device)
    intrinsics = sample['intrinsics'].unsqueeze(0).to(device)
    extrinsics = sample['extrinsics'].unsqueeze(0).to(device)

    # 4. Run Inference
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.float16):
            student_outputs = student(camera_image, intrinsics, extrinsics)
            
            # Handle output dict or raw tensor
            logits = student_outputs['bev_occupancy'] if isinstance(student_outputs, dict) else student_outputs
            
            # Convert logits to probabilities (0.0 to 1.0)
            pred_probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # 5. Generate Ground Truth for Comparison
    encoder = BEVGridEncoder(x_range=(0.0, 70.0), y_range=(-40.0, 40.0), bev_h=160, bev_w=160)
    bboxes = sample['bboxes'].unsqueeze(0)
    num_valid = sample['num_valid_boxes'].unsqueeze(0)
    encoded_targets = encoder.encode(bboxes, num_valid)
    gt_grid = encoded_targets['bev_occupancy'].squeeze().cpu().numpy()

    # 6. Visualization
    # Convert image from CHW [0, 1] to HWC for Matplotlib
    img_viz = sample['camera_image'].numpy().transpose(1, 2, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Student BEV Prediction - Validation Frame {frame_idx}", fontsize=16)

    # Plot 1: Camera Input
    axes[0].imshow(img_viz)
    axes[0].set_title("Front Camera Input")
    axes[0].axis('off')

    # Plot 2: Ground Truth BEV (Top: Far ahead, Bottom: Ego vehicle)
    axes[1].imshow(gt_grid, cmap='Blues', origin='upper')
    axes[1].set_title("Ground Truth BEV")
    axes[1].axis('off')

    # Plot 3: Predicted BEV Probabilities
    # Using 'magma' heatmap to show model confidence
    im = axes[2].imshow(pred_probs, cmap='magma', vmin=0.0, vmax=1.0, origin='upper')
    axes[2].set_title("Predicted BEV Heatmap")
    axes[2].axis('off')
    
    # Add a colorbar for the heatmap
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Occupancy Probability', rotation=270, labelpad=15)

    plt.tight_layout()
    save_path = 'prediction_viz.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")

if __name__ == '__main__':
    main()