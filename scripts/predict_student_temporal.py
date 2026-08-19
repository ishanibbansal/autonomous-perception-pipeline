import os
import sys
import glob
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Inject root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.perception.models.student_bev import StudentBEVDetector
from src.perception.utils.dataset import WaymoTemporalDataset, temporal_collate_fn
from src.perception.utils.target_encoder import BEVGridEncoder

def visualize_temporal(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing Temporal Visualizer on {device}...")

    # 1. Locate Data and Load Model
    val_files = glob.glob('data/raw/val/*.tfrecord')
    if not val_files:
        print("Error: No validation TFRecords found.")
        return

    dataset = WaymoTemporalDataset(val_files[0], is_train=False, seq_length=args.seq_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=temporal_collate_fn)

    student = StudentBEVDetector(use_temporal=True).to(device)
    print(f"Loading weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    student.load_state_dict(checkpoint['model_state_dict'])
    student.eval()

    encoder = BEVGridEncoder(x_range=(0.0, 70.0), y_range=(-40.0, 40.0), bev_h=160, bev_w=160)
    
    os.makedirs(args.out_dir, exist_ok=True)

    # 2. Inference Loop
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.num_samples:
                break
                
            past_features = None
            past_extrinsics = None
            
            # Step forward through time (e.g., t_2 -> t_1 -> t_0)
            for step in reversed(range(args.seq_length)):
                step_key = f't_{step}'
                step_data = batch[step_key]
                
                cam = step_data['camera_images'].to(device)
                intrin = step_data['intrinsics'].to(device)
                extrin = step_data['extrinsics'].to(device)
                extrin_inv = torch.inverse(extrin)
                
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    s_out = student(
                        cam, intrin, extrin_inv, 
                        past_features=past_features, 
                        current_extrinsics=extrin, 
                        past_extrinsics=past_extrinsics
                    )
                
                past_features = s_out['bev_features']
                past_extrinsics = extrin
                
                # Render visualization on the final frame (t_0)
                if step == 0:
                    # Extract raw probabilities from the network output
                    occ_logits = s_out['bev_occupancy']
                    occ_probs = torch.sigmoid(occ_logits)[0, 0].cpu().numpy()
                    
                    # Generate the ground truth from the LiDAR targets
                    encoded_targets = encoder.encode(step_data['bboxes'], step_data['num_valid_boxes'])
                    gt_occ = encoded_targets['bev_occupancy'][0, 0].cpu().numpy()
                    
                    # Process the image for Matplotlib
                    image_tensor = cam[0].cpu().numpy()
                    image_rgb = image_tensor.transpose(1, 2, 0)
                    image_rgb = np.clip(image_rgb, 0, 1)

                    # 3. Plotting
                    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
                    
                    axes[0].imshow(image_rgb)
                    axes[0].set_title(f'Front Camera (Frame t_0)', fontsize=14)
                    axes[0].axis('off')
                    
                    # Ground Truth (Crisp LiDAR)
                    axes[1].imshow(gt_occ, cmap='Blues', origin='lower', vmin=0, vmax=1)
                    axes[1].set_title('Ground Truth (LiDAR Boxes)', fontsize=14)
                    axes[1].axis('off')
                    
                    # Prediction (Camera Heatmap - Comet Tails)
                    axes[2].imshow(occ_probs, cmap='magma', origin='lower', vmin=0, vmax=0.8)
                    axes[2].set_title(f'Temporal Prediction (13.7% EMA)', fontsize=14)
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    out_path = os.path.join(args.out_dir, f'temporal_eval_sample_{batch_idx:03d}.png')
                    plt.savefig(out_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Saved visualization: {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Phase 2 Temporal Student')
    parser.add_argument('--checkpoint', type=str, default='best_temporal_student_checkpoint.pt')
    parser.add_argument('--seq_length', type=int, default=3, help='Must match training sequence length')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of random batches to visualize')
    parser.add_argument('--out_dir', type=str, default='runs/temporal_visualizations')
    args = parser.parse_args()
    
    visualize_temporal(args)