import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def render_occupancy_demo(raw_prediction, ground_truth, threshold=0.5):
    """
    Renders a side-by-side comparison of thresholded BEV predictions and ground truth.
    
    Args:
        raw_prediction (torch.Tensor or np.ndarray): The raw sigmoid probabilities (0.0 to 1.0)
        ground_truth (torch.Tensor or np.ndarray): The binary ground truth labels (0 or 1)
        threshold (float): The confidence cutoff to binarize the raw prediction.
    """
    # 1. Convert tensors to numpy arrays if they are PyTorch tensors
    if isinstance(raw_prediction, torch.Tensor):
        raw_prediction = raw_prediction.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
        
    # Ensure they are 2D grids (squeeze out batch/channel dimensions)
    raw_prediction = np.squeeze(raw_prediction)
    ground_truth = np.squeeze(ground_truth)

    # 2. Apply the binary threshold to create a crisp mask
    # This turns all probabilities > threshold into 1 (True), and the rest to 0 (False)
    binary_prediction = (raw_prediction > threshold).astype(np.uint8)

    # 3. Create a custom colormap for the demo
    # We will map 0 (empty road) to Black, and 1 (vehicle footprint) to a bright color
    pred_cmap = ListedColormap(['black', '#00FFCC']) # Neon cyan for predictions
    gt_cmap = ListedColormap(['black', '#39FF14'])   # Neon green for ground truth
    
    # 4. Set up the Matplotlib figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='#1E1E1E')
    fig.suptitle("WaymoBEVDetector: Occupancy Grid Evaluation", 
                 color='white', fontsize=16, fontweight='bold')

    # Plot Processed Prediction
    axes[0].imshow(binary_prediction, cmap=pred_cmap, interpolation='nearest')
    axes[0].set_title(f"Processed Prediction (Threshold: {threshold})", color='white')
    axes[0].axis('off') # Hide axes for a cleaner look

    # Plot Ground Truth
    axes[1].imshow(ground_truth, cmap=gt_cmap, interpolation='nearest')
    axes[1].set_title("Ground Truth (LiDAR-derived)", color='white')
    axes[1].axis('off')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
# Assuming you have `val_preds` from your model and `val_labels` from your dataloader:
# render_occupancy_demo(val_preds[0], val_labels[0], threshold=0.5)