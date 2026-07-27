import torch
from torch.utils.data import DataLoader

# Import your existing modules
from model import Waymo3DDetector
from utils.dataset import WaymoDataset
from utils.target_encoder import TargetEncoder

def debug_validation_batch(model, val_dataloader, encoder, device):
    print("\n--- RUNNING VALIDATION DEBUGGER ---")
    model.eval() # Ensure BatchNorm is using historical stats
    
    # Grab exactly ONE batch
    batch = next(iter(val_dataloader))
    
    images = batch['front_image'].to(device, dtype=torch.float32)
    raw_bboxes = batch['bboxes']
    valid_boxes = batch['num_valid_boxes']
    
    # Encode targets
    encoded_targets = encoder.encode(raw_bboxes, valid_boxes)
    targets = {k: v.to(device) for k, v in encoded_targets.items()}
    
    with torch.no_grad():
        predictions = model(images)
        
    # 1. Get the active mask and squeeze it to [Batch, H, W]
    active_mask = targets['mask'].bool().squeeze(1)
    
    # 2. Extract TARGET location and dimensions, permute, and apply mask
    target_loc = targets['location'].permute(0, 2, 3, 1)[active_mask]
    target_dim = targets['dimensions'].permute(0, 2, 3, 1)[active_mask]
    true_boxes = torch.cat([target_loc, target_dim], dim=-1) # Stitches into [X, Y, Z, L, W, H]
    
    # 3. Extract PREDICTED location and dimensions, permute, and apply mask
    pred_loc = predictions['location'].permute(0, 2, 3, 1)[active_mask]
    pred_dim = predictions['dimensions'].permute(0, 2, 3, 1)[active_mask]
    pred_boxes = torch.cat([pred_loc, pred_dim], dim=-1) # Stitches into [X, Y, Z, L, W, H]
    
    num_objects = true_boxes.shape[0]
    print(f"Found {num_objects} real vehicles in this validation batch.\n")
    
    for i in range(min(num_objects, 10)): # Print up to 10 cars
        print(f"Vehicle {i+1}:")
        # Formatting to 2 decimal places for clean reading
        true_fmt = [f"{val:.2f}" for val in true_boxes[i].tolist()]
        pred_fmt = [f"{val:.2f}" for val in pred_boxes[i].tolist()]
        
        print(f"  TARGET [X, Y, Z, L, W, H]: {true_fmt}")
        print(f"  PREDICT                  : {pred_fmt}")
        
        # Calculate depth error (assuming X is depth at index 0)
        depth_err = abs(true_boxes[i, 0] - pred_boxes[i, 0])
        print(f"  --> Depth Error: {depth_err.item():.2f} meters\n")
        
    print("--- DEBUG COMPLETE ---\n")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize the architecture
    model = Waymo3DDetector().to(device)
    encoder = TargetEncoder()
    
    # 2. Load your trained weights from the Epoch 15 checkpoint
    checkpoint_path = "waymo_3d_checkpoint.pt"
    print(f"Loading weights from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Depending on how you saved it, you might need checkpoint['model_state_dict']
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Could not find {checkpoint_path}. Ensure you are running this in the same directory as your training script.")
        exit(1)
        
    model.eval() # CRITICAL: Lock those BatchNorm layers!
    
    # 3. Initialize just the validation dataloader
    val_file = 'data/raw/segment-10072140764565668044_4060_000_4080_000_with_camera_labels.tfrecord'
    val_dataset = WaymoDataset(tfrecord_path=val_file)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 4. Fire the debugger
    debug_validation_batch(model, val_dataloader, encoder, device)