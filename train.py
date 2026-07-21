import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Waymo3DDetector
from loss import Waymo3DLoss
from utils.dataset import WaymoDataset
from utils.target_encoder import TargetEncoder

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
    
    # 3. Load the Real Waymo Data
    print("Loading Waymo .tfrecord Dataset...")
    tfrecord_file = 'data/raw/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
    dataset = WaymoDataset(tfrecord_path=tfrecord_file)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 4. Training Loop Variables
    epochs = 50
    checkpoint_path = "waymo_3d_checkpoint.pt"
    
    print(f"\nStarting {epochs}-epoch training run...\n")
    model.train() 
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # Extract inputs and raw targets
            images = batch['front_image'].to(device, dtype=torch.float32)
            raw_bboxes = batch['bboxes']
            valid_boxes = batch['num_valid_boxes']
            
            # Encode raw bounding boxes into dense spatial grids
            encoded_targets = encoder.encode(raw_bboxes, valid_boxes)
            
            # Move encoded targets to the GPU
            targets = {k: v.to(device) for k, v in encoded_targets.items()}
            
            # Zero out gradients
            optimizer.zero_grad()
            
            # Forward Pass
            predictions = model(images)
            
            # Calculate Loss
            loss, loss_components = criterion(predictions, targets)
            
            # Backward Pass & Optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Console Logging for Batches
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1:02d}/{epochs} | Batch {batch_idx} | Total Loss: {loss.item():.4f}")
        
        # 5. Checkpointing
        # Save a checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"--> Saved checkpoint at epoch {epoch + 1} to {checkpoint_path}")

if __name__ == '__main__':
    train_model()