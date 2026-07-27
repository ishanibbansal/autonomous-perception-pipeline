import os
import glob
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from model import Waymo3DDetector
from loss import Waymo3DLoss
from utils.dataset import WaymoDataset
from utils.target_encoder import TargetEncoder
from utils.validate import validate_model

def freeze_batchnorm(module):
    """Forces all BatchNorm layers to remain in evaluation mode."""
    classname = module.__class__.__name__
    if classname.find('BatchNorm') != -1:
        module.eval()

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing Training on: {device}")
    
    # Enable CuDNN benchmark for fixed image size optimization
    torch.backends.cudnn.benchmark = True

    model = Waymo3DDetector().to(device)
    criterion = Waymo3DLoss()
    encoder = TargetEncoder(image_width=960, image_height=640, grid_w=30, grid_h=20)
    

    # Differential learning rates: protecting the backbone features while training the head
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.head3d.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)
    
    print("Loading Waymo .tfrecord Datasets...")
    
    QUICK_DEBUG_RUN = False
    
    train_files = glob.glob('data/raw/train/*.tfrecord')
    if not train_files:
        raise FileNotFoundError("No .tfrecord files found in data/raw/train/")
        
    val_files = glob.glob('data/raw/val/*.tfrecord')
    if not val_files:
        raise FileNotFoundError("No .tfrecord files found in data/raw/val/")
        
    if QUICK_DEBUG_RUN:
        print("\n[INFO] Quick Debug Mode Enabled: Using 1 training file and 1 validation file.")
        train_files = train_files[:1]
        val_files = val_files[:1]
        epochs = 1
    else:
        epochs = 50

    print(f"Found {len(train_files)} training segments.")
    train_datasets = [WaymoDataset(tfrecord_path=f) for f in train_files]
    train_dataset = ConcatDataset(train_datasets)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    
    print(f"Found {len(val_files)} validation segments.")
    val_datasets = [WaymoDataset(tfrecord_path=f) for f in val_files]
    val_dataset = ConcatDataset(val_datasets)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Total training frames pooled: {len(train_dataset)}")
    print(f"Total validation frames pooled: {len(val_dataset)}")
    
    checkpoint_path = "waymo_3d_checkpoint.pt"
    best_checkpoint_path = "best_waymo_3d_checkpoint.pt"
    best_val_map = 0.0
    
    # Set how often to validate (e.g., every 2 epochs)
    VAL_INTERVAL = 2 
    
    print(f"\nStarting {epochs}-epoch training run...\n")
    
    for epoch in range(epochs):
        model.train() 
        model.apply(freeze_batchnorm) 
        epoch_train_loss = 0.0
        
        # Start the timer for the first batch
        batch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_dataloader):
            images = batch['front_image'].to(device, dtype=torch.float32)
            raw_bboxes = batch['bboxes']
            valid_boxes = batch['num_valid_boxes']
            
            encoded_targets = encoder.encode(raw_bboxes, valid_boxes)
            targets = {k: v.to(device) for k, v in encoded_targets.items()}
            
            # Vehicle Filter
            original_mask = targets['mask'].bool()
            vehicle_class_mask = (targets['class'][:, 0, :, :] == 1).unsqueeze(1)
            targets['mask'] = original_mask & vehicle_class_mask
            
            optimizer.zero_grad()
            predictions = model(images)
            loss, _ = criterion(predictions, targets)
            loss.backward()
            
            # Clip gradients to prevent exploding loss / NaN corruption
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            # Print stats every 10 batches
            if batch_idx % 10 == 0:
                # Calculate time elapsed
                elapsed_time = time.time() - batch_start_time
                batches_processed = 1 if batch_idx == 0 else 10
                sec_per_batch = elapsed_time / batches_processed
                
                print(f"Epoch {epoch + 1:02d}/{epochs} | Batch {batch_idx:03d} | Loss: {loss.item():.4f} | Speed: {sec_per_batch:.3f} sec/batch")
                
                # Reset the timer for the next 10 batches
                batch_start_time = time.time()
                
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        
        # --- NEW VALIDATION LOGIC ---
        # Only run validation every VAL_INTERVAL epochs, or on the very last epoch
        if (epoch + 1) % VAL_INTERVAL == 0 or (epoch + 1) == epochs or QUICK_DEBUG_RUN:
            
            # Raised conf_thresh to 0.25 to speed up NMS processing
            avg_val_loss, avg_map = validate_model(model, val_dataloader, criterion, encoder, device, conf_thresh=0.25)
            
            print(f"Epoch {epoch + 1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mAP: {avg_map:.4f}")
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_map': avg_map,
            }
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0 or QUICK_DEBUG_RUN:
                torch.save(checkpoint, checkpoint_path)
                print(f"--> Saved periodic checkpoint at epoch {epoch + 1} to {checkpoint_path}")
                
            # Save best checkpoint if mAP improves
            if avg_map > best_val_map:
                best_val_map = avg_map
                torch.save(checkpoint, best_checkpoint_path)
                print(f"--> [NEW BEST] Saved peak model with Val mAP: {avg_map:.4f} to {best_checkpoint_path}")
                
        else:
            # On skipped epochs, just print the training loss
            print(f"Epoch {epoch + 1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Validation Skipped (Speed Up)")

if __name__ == '__main__':
    train_model()