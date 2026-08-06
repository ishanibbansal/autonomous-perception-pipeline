import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import time
import argparse
import warnings
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from model import WaymoBEVDetector
from loss import WaymoDetectionLoss
from utils.dataset import WaymoDataset, waymo_collate_fn
from utils.target_encoder import BEVGridEncoder
from utils.validate import validate_model

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing LiDAR-Camera Sensor Fusion Training on: {device}")
    
    torch.backends.cudnn.benchmark = True

    writer = SummaryWriter(log_dir='runs/lidar_camera_teacher_01')
    print("TensorBoard initialized. Run 'tensorboard --logdir=runs' to view.")

    # --- Initialize Model ---
    model = WaymoBEVDetector().to(device)

    # --- NEW: Initialize the Master Pipeline Loss ---
    criterion = WaymoDetectionLoss().to(device)
    
    encoder = BEVGridEncoder(x_range=(0.0, 70.0), y_range=(-40.0, 40.0), bev_h=160, bev_w=160)
    
    epochs = 15
    ACCUMULATION_STEPS = 4 
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)                                               
    
    warmup_epochs = 2
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-8)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    scaler = torch.amp.GradScaler('cuda')
    
    print("Loading Waymo .tfrecord Datasets...")
    
    train_files = glob.glob('data/raw/train/*.tfrecord')
    if not train_files:
        raise FileNotFoundError("No .tfrecord files found in data/raw/train/")
        
    val_files = glob.glob('data/raw/val/*.tfrecord')
    if not val_files:
        raise FileNotFoundError("No .tfrecord files found in data/raw/val/")

    # Note: Adjust num_sweeps here if needed for CPU testing
    train_datasets = [WaymoDataset(tfrecord_path=f, is_train=True) for f in train_files]
    train_dataset = ConcatDataset(train_datasets)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=2,              
        pin_memory=True, 
        collate_fn=waymo_collate_fn,
        persistent_workers=True,     
        prefetch_factor=2            
    )
    
    val_datasets = [WaymoDataset(tfrecord_path=f, is_train=False) for f in val_files]
    val_dataset = ConcatDataset(val_datasets)
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=4,             # Bump back up to 4 to cut total batches in half
        shuffle=False, 
        num_workers=2,            # Use 2 workers safely since validation has no gradients/backward pass
        pin_memory=True, 
        collate_fn=waymo_collate_fn,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    checkpoint_path = "waymo_bev_checkpoint.pt"
    best_checkpoint_path = "best_waymo_bev_checkpoint.pt"
    best_val_score = 0.0
    start_epoch = 0
    VAL_INTERVAL = 3

    if args.resume and os.path.exists(best_checkpoint_path):
        print(f"Found existing checkpoint at {best_checkpoint_path}. Resuming training...")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch']
        best_val_score = checkpoint.get('val_score', 0.0)
        print(f"Resumed successfully! Starting from Epoch {start_epoch + 1} with previous Best Score: {best_val_score:.4f}")
    elif args.resume:
        print(f"Warning: --resume flag passed, but no checkpoint found at {best_checkpoint_path}. Starting from scratch.")
    
    print(f"\nStarting training run from epoch {start_epoch + 1} to {epochs}...\n")
    
    for epoch in range(start_epoch, epochs):
        model.train() 
        
        epoch_train_loss = 0.0
        batch_start_time = time.time()
        
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(train_dataloader):
            lidar_points = batch['lidar_points'].to(device, non_blocking=True)
            batch_indices = batch['batch_indices'].to(device, non_blocking=True)
            
            camera_images = batch['camera_images'].to(device, non_blocking=True)
            lidar_uvs = batch['lidar_uvs'].to(device, non_blocking=True)
            
            # --- Move Encoded Targets to Device Safely ---
            encoded_targets = encoder.encode(batch['bboxes'], batch['num_valid_boxes'])
            targets_gpu = {k: v.to(device, non_blocking=True) for k, v in encoded_targets.items()}
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                predictions = model(lidar_points, batch_indices, camera_images, lidar_uvs)
                
                # --- The Master Loss Wrapper applies automatically ---
                total_loss = criterion(predictions, targets_gpu)
                loss = total_loss / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                optimizer.zero_grad()
            
            true_loss = loss.item() * ACCUMULATION_STEPS
            epoch_train_loss += true_loss
            
            global_step = epoch * len(train_dataloader) + batch_idx
            writer.add_scalar('Training/Batch_Loss', true_loss, global_step)
            
            if batch_idx % 500 == 0:
                with torch.no_grad():
                    pred_grid = torch.sigmoid(predictions['bev_occupancy'][0:1]) 
                    target_grid = targets_gpu['bev_occupancy'][0:1]
                    writer.add_image('BEV/1_Prediction', pred_grid[0], global_step)
                    writer.add_image('BEV/2_Ground_Truth', target_grid[0], global_step)
            
            if batch_idx % 10 == 0:
                elapsed_time = time.time() - batch_start_time
                batches_processed = 1 if batch_idx == 0 else 10
                sec_per_batch = elapsed_time / batches_processed
                print(f"Epoch {epoch + 1:02d}/{epochs} | Batch {batch_idx:03d} | Loss: {true_loss:.4f} | Speed: {sec_per_batch:.3f} sec/batch")
                batch_start_time = time.time()
                
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            scheduler.step()
        
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        writer.add_scalar('Training/Epoch_Loss', avg_train_loss, epoch)
        writer.add_scalar('Training/Learning_Rate_Head', optimizer.param_groups[0]['lr'], epoch)
        
        if (epoch + 1) % VAL_INTERVAL == 0 or (epoch + 1) == epochs:
            avg_val_loss, avg_score = validate_model(model, val_dataloader, criterion, encoder, device)
            
            writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
            writer.add_scalar('Validation/Score', avg_score, epoch)
            
            print(f"Epoch {epoch + 1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Score: {avg_score:.4f}")
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_score': avg_score,
            }
            
            if (epoch + 1) % 5 == 0:
                torch.save(checkpoint, checkpoint_path)
                print(f"--> Saved periodic checkpoint at epoch {epoch + 1} to {checkpoint_path}")
                
            if avg_score > best_val_score:
                best_val_score = avg_score
                torch.save(checkpoint, best_checkpoint_path)
                print(f"--> [NEW BEST] Saved peak model with Val Score: {avg_score:.4f} to {best_checkpoint_path}")
                
        else:
            print(f"Epoch {epoch + 1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Validation Skipped")

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Waymo BEV Detector')
    parser.add_argument('--resume', action='store_true', help='Resume training from best checkpoint')
    args = parser.parse_args()
    
    mp.set_start_method('spawn', force=True)
    train_model(args)