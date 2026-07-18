import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import WaymoDataset  # Importing the class we just built

def validate_pipeline(batch):
    print("Generating validation plot...")
    
    # 1. Extract the very first frame from our batch of 4
    image_tensor = batch['front_image'][0]
    boxes_tensor = batch['bboxes'][0]
    valid_count = batch['num_valid_boxes'][0].item()

    # 2. Convert PyTorch Image [C, H, W] back to standard [H, W, C] for visualization
    image_np = image_tensor.permute(1, 2, 0).numpy().astype(np.uint8)

    # 3. Slice out the zero-padding! We only want the real boxes
    valid_boxes = boxes_tensor[:valid_count].numpy()

    # 4. Setup the Side-by-Side Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- Plot 1: Front Camera ---
    ax1.imshow(image_np)
    ax1.set_title("Pipeline Output: Front Camera Image")
    ax1.axis('off')

    # --- Plot 2: Bird's Eye View (BEV) of Targets ---
    # In Waymo's coordinate frame: X is forward, Y is left/right
    x_coords = valid_boxes[:, 1]
    y_coords = valid_boxes[:, 2]
    
    ax2.scatter(y_coords, x_coords, c='red', marker='s', label='Detected Objects')
    ax2.plot(0, 0, 'b^', markersize=12, label='Ego Vehicle (You)') # The center of the map
    
    ax2.set_xlim(-40, 40)  # 40 meters left/right
    ax2.set_ylim(-20, 80)  # 20m behind, 80m ahead
    ax2.set_title(f"Pipeline Output: 3D Targets (Bird's Eye View)\nTotal Real Targets: {valid_count}")
    ax2.set_xlabel("Y (Meters Left/Right)")
    ax2.set_ylabel("X (Meters Forward/Backward)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('outputs/validation_plot.png', dpi=300)
    print("Plot saved as validation_plot.png! Check your VS Code file explorer.")

if __name__ == '__main__':
    data_path = 'data/raw/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
    
    print("Loading dataset...")
    dataset = WaymoDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Grab the first batch and validate it
    for batch in dataloader:
        validate_pipeline(batch)
        break