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
from loss import BEVFocalLoss, CombinedBEVLoss
from utils.dataset import WaymoDataset
from utils.target_encoder import BEVGridEncoder

def freeze_batchnorm(module):
    """Forces all BatchNorm layers to remain in evaluation mode."""
    classname = module.__class__.__name__
    if classname.find('BatchNorm') != -1:
        module.eval()

def custom_collate_fn(batch):
    return {
        'timestamp': torch.stack([item['timestamp'] for item in batch]),
        'front_image': torch.stack([item['front_image'] for item in batch]),
        'bboxes': torch.stack([item['bboxes'] for item in batch]),
        'num_valid_boxes': torch.stack([item['num_valid_boxes'] for item in batch]),
        'lidar_points': [item['lidar_points'] for item in batch]
    }

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing Fully Optimized Training on: {device}")
    
    torch.backends.cudnn.benchmark = True

    # 1. Initialize TensorBoard Writer (New directory for fresh run)
    writer = SummaryWriter(log_dir='runs/bev_experiment_04')
    print("TensorBoard initialized. Run 'tensorboard --logdir=runs' to view.")

    # 2. Setup Architecture & Combined Loss (Focal + Soft-IoU)
    model = WaymoBEVDetector().to(device)
    base_focal = BEVFocalLoss(alpha=0.25, gamma=2.0)
    criterion = CombinedBEVLoss(focal_loss=base_focal, iou_weight=2.0).to(device)
    encoder = BEVGridEncoder(x_range=(0.0, 80.0), y_range=(-40.0, 40.0), resolution=0.5)
    
    epochs = 50
    ACCUMULATION_STEPS = 4 # Simulates batch size 16 (4 batch * 4 steps)
    
    # REDUCED HEAD LR AND INCREASED WEIGHT DECAY
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.bev_head.parameters(), 'lr': 5e-4}, # Lowered from 1e-3
    ], weight_decay=2e-3) # Doubled from 1e-3
    
    # LEARNING RATE WARMUP + COSINE DECAY
    warmup_epochs = 2
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-8)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    # AMP Scaler for mixed precision training (RTX 2060 optimization)
    scaler = torch.amp.GradScaler('cuda')
    
    print("Loading Waymo .tfrecord Datasets...")
    
    train_files = glob.glob('data/raw/train/*.tfrecord')
    if not train_files:
        raise FileNotFoundError("No .tfrecord files found in data/raw/train/")
        
    val_files = glob.glob('data/raw/val/*.tfrecord')
    if not val_files:
        raise FileNotFoundError("No .tfrecord files found in data/raw/val/")

    train_datasets = [WaymoDataset(tfrecord_path=f) for f in train_files]
    train_dataset = ConcatDataset(train_datasets)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=2,               
        pin_memory=True, 
        collate_fn=custom_collate_fn,
        persistent_workers=True,     
        prefetch_factor=2            
    )
    
    val_datasets = [WaymoDataset(tfrecord_path=f) for f in val_files]
    val_dataset = ConcatDataset(val_datasets)
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True, 
        collate_fn=custom_collate_fn,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    checkpoint_path = "waymo_bev_checkpoint.pt"
    best_checkpoint_path = "best_waymo_bev_checkpoint.pt"
    best_val_score = 0.0
    start_epoch = 0
    VAL_INTERVAL = 2 

    # 3. Handle Checkpoint Resuming
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
        model.backbone.apply(freeze_batchnorm) 
        
        epoch_train_loss = 0.0
        batch_start_time = time.time()
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_dataloader):
            images = batch['front_image'].to(device, dtype=torch.float32, non_blocking=True)
            # Normalize only the 3 RGB channels (leave LiDAR depth in true meters)
            images[:, :3] = images[:, :3] / 255.0
            
            raw_bboxes = batch['bboxes']
            valid_boxes = batch['num_valid_boxes']
            
            encoded_targets = encoder.encode(raw_bboxes, valid_boxes)
            targets = encoded_targets['bev_occupancy'].to(device, non_blocking=True)
            
            # Execute forward pass with Automatic Mixed Precision
            with torch.amp.autocast('cuda'):
                predictions = model(images)
                loss = criterion(predictions['bev_occupancy'], targets)
                # Scale loss by accumulation steps
                loss = loss / ACCUMULATION_STEPS
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            
            # Step optimizer and update weights after accumulating N steps
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
            
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    pred_grid = torch.sigmoid(predictions['bev_occupancy'][0:1]) 
                    target_grid = targets[0:1]
                    writer.add_image('BEV/1_Prediction', pred_grid[0], global_step)
                    writer.add_image('BEV/2_Ground_Truth', target_grid[0], global_step)
            
            if batch_idx % 10 == 0:
                elapsed_time = time.time() - batch_start_time
                batches_processed = 1 if batch_idx == 0 else 10
                sec_per_batch = elapsed_time / batches_processed
                print(f"Epoch {epoch + 1:02d}/{epochs} | Batch {batch_idx:03d} | Loss: {true_loss:.4f} | Speed: {sec_per_batch:.3f} sec/batch")
                batch_start_time = time.time()
                
        # Step the learning rate scheduler at the end of the epoch (suppressing PyTorch SequentialLR internal warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            scheduler.step()
        
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        writer.add_scalar('Training/Epoch_Loss', avg_train_loss, epoch)
        writer.add_scalar('Training/Learning_Rate_Head', optimizer.param_groups[1]['lr'], epoch)
        
        if (epoch + 1) % VAL_INTERVAL == 0 or (epoch + 1) == epochs:
            from utils.validate import validate_model
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

        torch.cuda.empty_cache()

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Waymo BEV Detector')
    parser.add_argument('--resume', action='store_true', help='Resume training from best checkpoint')
    args = parser.parse_args()
    
    mp.set_start_method('spawn', force=True)
    train_model(args)