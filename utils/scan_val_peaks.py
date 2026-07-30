import os
import sys

# Anchor paths to the project root directory (one level up from utils)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import glob
from torch.utils.data import ConcatDataset

from model import WaymoBEVDetector 
from utils.dataset import WaymoDataset

def scan_validation_peaks(checkpoint_path='best_waymo_bev_checkpoint.pt', num_frames=50):
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(ROOT_DIR, checkpoint_path)
        
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
        for key, value in base_state.items():
            if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                ema_key = 'module.' + key 
                if ema_key in ema_state:
                    ema_state[ema_key] = value
        state_dict = ema_state
    else:
        state_dict = base_state if base_state else checkpoint

    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=False)

    model.to(device)
    model.eval()
    
    # --- SETUP VALIDATION DATASET ---
    val_pattern = os.path.join(ROOT_DIR, 'data', 'raw', 'val', '*.tfrecord')
    val_files = glob.glob(val_pattern)
    if not val_files:
        val_files = glob.glob('data/raw/val/*.tfrecord')
        
    if not val_files:
        raise FileNotFoundError("No .tfrecord files found in data/raw/val/")

    val_datasets = [WaymoDataset(tfrecord_path=f) for f in val_files]
    val_dataset = ConcatDataset(val_datasets)
    
    print(f"\nScanning first {num_frames} validation frames for peak confidence outputs...\n")
    print(f"{'Frame Index':<12} | {'GT Vehicles':<12} | {'Peak Probability':<18}")
    print("-" * 50)
    
    scanned_count = min(num_frames, len(val_dataset))
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for idx in range(scanned_count):
            sample = val_dataset[idx]
            images = sample['front_image'].unsqueeze(0).to(device, dtype=torch.float32)
            num_boxes = sample['num_valid_boxes'].item()
            
            predictions = model(images)
            pred_prob = torch.sigmoid(predictions['bev_occupancy'])
            peak_val = pred_prob.max().item()
            
            print(f"{idx:<12} | {num_boxes:<12} | {peak_val:<18.4f}")

if __name__ == '__main__':
    checkpoint_file = 'best_waymo_bev_checkpoint.pt'
    scan_validation_peaks(checkpoint_path=checkpoint_file, num_frames=50)