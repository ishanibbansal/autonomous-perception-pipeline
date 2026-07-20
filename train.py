import torch
import torch.optim as optim
from model import Waymo3DDetector
from loss import Waymo3DLoss

def train_single_batch():
    print("Initializing Model and Loss...")
    # 1. Instantiate the architecture and multi-task loss function
    model = Waymo3DDetector()
    criterion = Waymo3DLoss()
    
    # 2. Set up the Optimizer
    # We use Adam with a standard learning rate (1e-3) for rapid, stable convergence
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Create the Dummy Data (The Single Batch)
    print("Generating single-batch data for the overfitting test...")
    dummy_input = torch.randn(1, 3, 1280, 1920)
    
    # Ground truth targets must perfectly match the grid shape of the outputs (40x60)
    dummy_targets = {
        'class': torch.rand(1, 3, 40, 60),
        'location': torch.randn(1, 3, 40, 60),
        'dimensions': torch.randn(1, 3, 40, 60),
        'orientation': torch.randn(1, 2, 40, 60)
    }
    
    # 4. The Overfitting Loop
    epochs = 50
    print(f"\nStarting {epochs}-epoch single-batch overfitting run...\n")
    
    model.train() # Set the model to training mode
    
    for epoch in range(epochs):
        # A. Zero out the gradients from the previous iteration
        optimizer.zero_grad()
        
        # B. Forward Pass: Push the image through the network
        predictions = model(dummy_input)
        
        # C. Calculate the multi-task loss
        loss, loss_components = criterion(predictions, dummy_targets)
        
        # D. Backward Pass: Calculate the gradients 
        loss.backward()
        
        # E. Update the network's weights based on the gradients
        optimizer.step()
        
        # F. Console Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:02d}/{epochs} | Total Loss: {loss.item():.4f} | "
                  f"Cls: {loss_components['cls_loss']:.4f}, "
                  f"Loc: {loss_components['loc_loss']:.4f}, "
                  f"Dim: {loss_components['dim_loss']:.4f}, "
                  f"Ori: {loss_components['orient_loss']:.4f}")

if __name__ == '__main__':
    train_single_batch()