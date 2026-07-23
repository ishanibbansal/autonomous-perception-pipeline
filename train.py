import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Waymo3DDetector
from loss import Waymo3DLoss
from utils.dataset import WaymoDataset
from utils.target_encoder import TargetEncoder
from utils.validate import validate_model

def train_model():
    # Setup GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing Training on: {device}")

    # 1. Instantiate Architecture, Loss, and Encoder
    model = Waymo3DDetector().to(device)
    criterion = Waymo3DLoss()
    encoder = TargetEncoder()
    
    # 2. Set up Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Load the Real Waymo Data (Train & Val)
    print("Loading Waymo .tfrecord Datasets...")
    train_file = 'data/raw/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
    val_file = 'data/raw/segment-10072140764565668044_4060_000_4080_000_with_camera_labels.tfrecord'
    
    train_dataset = WaymoDataset(tfrecord_path=train_file)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    val_dataset = WaymoDataset(tfrecord_path=val_file)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False) # No need to shuffle validation data
    
    # 4. Training Loop Variables
    epochs = 50
    checkpoint_path = "waymo_3d_checkpoint.pt"
    
    print(f"\nStarting {epochs}-epoch training run...\n")
    
    for epoch in range(epochs):
        model.train() # Set to train mode at the start of every epoch
        epoch_train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_dataloader):
            images = batch['front_image'].to(device, dtype=torch.float32)
            raw_bboxes = batch['bboxes']
            valid_boxes = batch['num_valid_boxes']
            
            encoded_targets = encoder.encode(raw_bboxes, valid_boxes)
            targets = {k: v.to(device) for k, v in encoded_targets.items()}
            
            optimizer.zero_grad()
            predictions = model(images)
            loss, _ = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            # ---> BATCH LOGGING: Print an update every 10 batches <---
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1:02d}/{epochs} | Batch {batch_idx:03d} | Loss: {loss.item():.4f}")
                
        # Calculate average training loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        
        # 5. Validation Loop
        avg_val_loss, avg_map = validate_model(model, val_dataloader, criterion, encoder, device)
        
        # 6. Epoch Logging (Cleaned up duplicates)
        print(f"Epoch {epoch + 1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mAP: {avg_map:.4f}")
        
        # 7. Checkpointing
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_map': avg_map,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"--> Saved checkpoint at epoch {epoch + 1} to {checkpoint_path}")

if __name__ == '__main__':
    train_model()