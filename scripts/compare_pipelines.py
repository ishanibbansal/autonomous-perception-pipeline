import sys
import os
import argparse
import glob
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Inject repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)

from src.perception.models.student_bev import StudentBEVDetector
from src.perception.utils.dataset import WaymoDataset
from src.perception.utils.target_encoder import BEVGridEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Run Python BEV inference and export matching C++ test frame & calib.")
    parser.add_argument("--tfrecord", type=str, default=None, help="Path to TFRecord file (default: first found in val or train)")
    parser.add_argument("--frame-idx", type=int, default=None, help="Frame index to use (default: first frame with >= 3 boxes)")
    parser.add_argument("--checkpoint", type=str, default="best_student_bev_checkpoint.pt", help="Path to student checkpoint")
    parser.add_argument("--output-img", type=str, default="test_frame_clean.png", help="Output path for lossless test image")
    return parser.parse_args()


def find_tfrecord():
    val_files = sorted(glob.glob(os.path.join(REPO_ROOT, 'data/raw/val/*.tfrecord')))
    if val_files:
        return val_files[0]
    train_files = sorted(glob.glob(os.path.join(REPO_ROOT, 'data/raw/train/*.tfrecord')))
    if train_files:
        return train_files[0]
    all_files = sorted(glob.glob(os.path.join(REPO_ROOT, 'data/raw/*.tfrecord')))
    if all_files:
        return all_files[0]
    raise FileNotFoundError("No .tfrecord files found under data/raw/val/ or data/raw/train/")


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Pipeline Parity & Comparison Tool ===")
    print(f"Running on device: {device}")

    # 1. Resolve TFRecord
    tfrecord_path = args.tfrecord if args.tfrecord else find_tfrecord()
    print(f"Using TFRecord: {tfrecord_path}")

    # 2. Load Dataset
    dataset = WaymoDataset(tfrecord_path=tfrecord_path, is_train=False, num_sweeps=1)
    print(f"Total frames in dataset: {len(dataset)}")

    # 3. Pick Frame
    if args.frame_idx is not None:
        frame_idx = args.frame_idx
        sample = dataset[frame_idx]
    else:
        # Search for a frame with multiple valid boxes for a clear BEV detection
        frame_idx = 0
        for idx in range(len(dataset)):
            s = dataset[idx]
            if s['num_valid_boxes'] >= 3:
                frame_idx = idx
                sample = s
                break
        else:
            sample = dataset[0]
            frame_idx = 0

    print(f"Selected frame index: {frame_idx} (valid boxes: {sample['num_valid_boxes'].item()})")

    # 4. Save lossless 1280x1920 test frame (in RGB -> BGR for OpenCV / ROS)
    raw_cam_img = sample['camera_image'].numpy().transpose(1, 2, 0) # [1280, 1920, 3] in [0, 1]
    img_uint8 = (raw_cam_img * 255.0).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    out_img_path = os.path.join(REPO_ROOT, args.output_img)
    cv2.imwrite(out_img_path, img_bgr)
    print(f"Saved lossless 1280x1920 test frame to: {out_img_path}")

    # 5. Extract Intrinsics & Extrinsics
    intrinsics = sample['intrinsics'].numpy()
    extrinsics = sample['extrinsics'].numpy()

    print("\n" + "=" * 50)
    print("// --- COPY THIS INTO YOUR C++ NODE ---")
    print("// 1. Front Camera Intrinsics")
    print("float intrinsics[9] = {")
    print(f"    {intrinsics[0, 0]:.6f}f, 0.0f, {intrinsics[0, 2]:.6f}f,")
    print(f"    0.0f, {intrinsics[1, 1]:.6f}f, {intrinsics[1, 2]:.6f}f,")
    print("    0.0f, 0.0f, 1.0f")
    print("};")

    print("\n// 2. Front Camera Extrinsics RAW (Sensor to Vehicle)")
    print("float extrinsics_inv[16] = {")
    for row in extrinsics:
        print(f"    {row[0]:.6f}f, {row[1]:.6f}f, {row[2]:.6f}f, {row[3]:.6f}f,")
    print("};")
    print("=" * 50 + "\n")

    # 6. Load Student Model
    ckpt_path = os.path.join(REPO_ROOT, args.checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find checkpoint: {ckpt_path}")

    student = StudentBEVDetector().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    student.load_state_dict(checkpoint['model_state_dict'])
    student.eval()
    print(f"Loaded student model from checkpoint: {ckpt_path} (epoch {checkpoint.get('epoch', 'N/A')})")

    # 7. Run PyTorch Inference (both FP32 and FP16 to check sensitivity)
    cam_tensor = sample['camera_image'].unsqueeze(0).to(device)
    intr_tensor = sample['intrinsics'].unsqueeze(0).to(device)
    extr_tensor = sample['extrinsics'].unsqueeze(0).to(device)

    with torch.no_grad():
        # FP32 inference (matches default ONNX / TensorRT precision)
        out_fp32 = student(cam_tensor, intr_tensor, extr_tensor)
        logits_fp32 = out_fp32['bev_occupancy'] if isinstance(out_fp32, dict) else out_fp32
        probs_fp32 = torch.sigmoid(logits_fp32).squeeze().cpu().numpy()

        # FP16 inference (matches predict_student.py autocast)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out_fp16 = student(cam_tensor, intr_tensor, extr_tensor)
            logits_fp16 = out_fp16['bev_occupancy'] if isinstance(out_fp16, dict) else out_fp16
            probs_fp16 = torch.sigmoid(logits_fp16).squeeze().cpu().numpy()

    print("\n=== Python Model Statistics ===")
    print(f"FP32 Probabilities - Min: {probs_fp32.min():.4f}, Max: {probs_fp32.max():.4f}, Mean: {probs_fp32.mean():.4f}")
    print(f"FP16 Probabilities - Min: {probs_fp16.min():.4f}, Max: {probs_fp16.max():.4f}, Mean: {probs_fp16.mean():.4f}")
    diff = np.abs(probs_fp32 - probs_fp16)
    print(f"Max absolute diff between FP32 and FP16: {diff.max():.6f}")

    # 8. Generate Ground Truth BEV
    encoder = BEVGridEncoder(x_range=(0.0, 70.0), y_range=(-40.0, 40.0), bev_h=160, bev_w=160)
    bboxes = sample['bboxes'].unsqueeze(0)
    num_valid = sample['num_valid_boxes'].unsqueeze(0)
    encoded_targets = encoder.encode(bboxes, num_valid)
    gt_grid = encoded_targets['bev_occupancy'].squeeze().cpu().numpy()

    # 9. Save Matplotlib 3-panel Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Student BEV Comparison - Frame {frame_idx}", fontsize=16)

    axes[0].imshow(raw_cam_img)
    axes[0].set_title("Front Camera Input (1280x1920)")
    axes[0].axis('off')

    axes[1].imshow(gt_grid, cmap='Blues', origin='upper')
    axes[1].set_title("Ground Truth BEV (Top: Far, Bottom: Ego)")
    axes[1].axis('off')

    im = axes[2].imshow(probs_fp32, cmap='magma', vmin=0.0, vmax=1.0, origin='upper')
    axes[2].set_title(f"PyTorch Predicted BEV (Max Conf: {probs_fp32.max():.2f})")
    axes[2].axis('off')
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Occupancy Probability', rotation=270, labelpad=15)

    plt.tight_layout()
    viz_path = os.path.join(REPO_ROOT, 'python_prediction_viz.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PyTorch 3-panel visualization to: {viz_path}")

    # 10. Save OpenCV-equivalent Heatmap (matching C++ post-processing exactly)
    # Row 0 = Top (X=70m), Row 159 = Bottom (X=0m). No flip needed.
    heatmap_u8 = (probs_fp32 * 255.0).clip(0, 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_MAGMA)

    # Resize display image down to 960x640 and heatmap to 640x640, matching C++
    disp_img = cv2.resize(img_bgr, (960, 640))
    disp_heat = cv2.resize(heatmap_colored, (640, 640))
    combined = cv2.hconcat([disp_img, disp_heat])

    cv_out_path = os.path.join(REPO_ROOT, 'python_cpp_equivalent_output.jpg')
    cv2.imwrite(cv_out_path, combined)
    print(f"Saved C++ format-matched reference image to: {cv_out_path}")
    print("\nDone! Next, update C++ node with the calibration above and test_ros_node.py to use test_frame_clean.png.")


if __name__ == '__main__':
    main()
