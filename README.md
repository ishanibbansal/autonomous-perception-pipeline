# Waymo BEV Perception Pipeline: Cross-Modal Distillation

A 2D/3D autonomous vehicle perception pipeline trained on the Waymo Open Dataset. This repository implements a **Teacher-Student Knowledge Distillation** architecture to transfer forward-facing spatial geometry (a 70x80 meter physical grid) from a heavy, multi-modal sensor fusion model (LiDAR + Camera) into a lightweight, vision-only monocular model (Camera only).

## Features

* **Multi-Modal Teacher Network:** Fuses historical LiDAR sweeps (`num_sweeps=3`) using a PointPillars encoder and a ResNet perspective camera backbone via a BiFPN Bird's-Eye View (BEV) decoder.
* **Monocular Student Network:** A lightweight, camera-only architecture trained to mimic the Teacher's intermediate semantic feature maps and dense occupancy grids within a 70m deep by 80m wide forward-facing spatial grid (`x_range=(0.0, 70.0)`, `y_range=(-40.0, 40.0)`).
* **CenterNet Spatial Encoding:** Replaces rigid binary grids with continuous 2D Gaussian heatmaps (160x160 resolution) for stable gradient optimization. A strict forward-FOV filter (`X > 2.0`) automatically excludes non-visible rear geometry.
* **Hardware-Optimized Training:** Designed to train dual deep neural networks simultaneously on hardware constrained to 6GB of VRAM (e.g., NVIDIA RTX 2060) utilizing strict gradient accumulation and optimized multiprocessing.

---

## System Architecture & Network Setup

This project uses a distributed headless compute model:
* **Host (Compute Engine):** Windows PC running Ubuntu via WSL2.
* **Client (Control Center):** macOS running VS Code Remote-SSH and Foxglove Studio.
* **Networking:** Zero-configuration secure tunnel via Tailscale (`100.x.x.x` subnet).

**WSL2 GPU Setup:**
Ensure your WSL2 environment can access the host NVIDIA drivers by adding this to your `~/.bashrc`:

    export PATH=$PATH:/usr/lib/wsl/lib

---

## Directory Structure

    ├── data/
    │   └── raw/
    │       ├── train/          # Waymo .tfrecord training files
    │       └── val/            # Waymo .tfrecord validation files
    ├── docs/                   # Documentation and engineering logs
    │   └── dev_log.md
    ├── notebooks/              # Jupyter notebooks for EDA
    │   └── .gitkeep
    ├── scripts/                # Utility scripts
    │   └── .gitkeep
    ├── src/                    # ROS 2 integration and node source code
    │   ├── perception/
    │   │   └── .gitkeep
    │   └── prediction/
    │       └── .gitkeep
    ├── utils/                  # Helper modules and data pipelines
    │   ├── dataset.py          # Data parsing, temporal sweeps, sensor fusion projection
    │   ├── download_dataset.py # Waymo data fetcher
    │   ├── metrics.py          # Validation metric calculations
    │   ├── parser.py           # TFRecord parser
    │   ├── scan_val_peaks.py   # Peak extraction visualizer
    │   ├── target_encoder.py   # CenterNet BEV grid target generation
    │   ├── validate.py         # Inline Grid IoU and validation logic
    │   └── visualization.py    # TensorBoard hooks
    ├── .gitignore              # Git ignore file
    ├── README.md               # Project documentation
    ├── debug.py                # Standalone debugging scripts
    ├── distill_loss.py         # CrossModalDistillationLoss for Student
    ├── loss.py                 # Multi-task objective loss for Teacher
    ├── model.py                # WaymoBEVDetector (Multi-Modal Teacher)
    ├── predict.py              # Baseline inference script
    ├── predict_bev.py          # Inference and Side-by-Side Validation Visualizer
    ├── predict_cpu.py          # CPU-only inference fallback
    ├── student_model.py        # StudentBEVDetector (Monocular Student)
    ├── test_frame.py           # Single frame dataloader test
    ├── train.py                # Teacher training pipeline
    └── train_student.py        # Student distillation pipeline

---

## Installation

1. **Clone the repository:**

        git clone https://github.com/ishanibbansal/autonomous-perception-pipeline.git
        cd autonomous-perception-pipeline

2. **Install Dependencies:**
   Ensure you have PyTorch (with CUDA support) and TensorFlow (strictly for parsing `.tfrecord` files). 

        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install tensorflow opencv-python matplotlib tensorboard

3. **Data Preparation:**
   Download the Waymo Open Dataset `.tfrecord` files and place them in the correct directories:
   * `data/raw/train/*.tfrecord`
   * `data/raw/val/*.tfrecord`

---

## Usage

### 1. Train the Teacher Model (LiDAR + Camera)
The Teacher must be trained first to generate the spatial targets. It is recommended to use `nohup` to run the training continuously in the background:

    nohup python -u train.py > training.log 2>&1 &

To resume training from the last saved checkpoint, append the `--resume` flag:

    nohup python -u train.py --resume > training.log 2>&1 &

*Outputs:* `best_waymo_bev_checkpoint.pt`

### 2. Train the Student Model (Camera Only via Distillation)
Once the Teacher is trained, freeze its weights and begin cross-modal distillation. This script runs both models simultaneously.

    nohup python -u train_student.py > training.log 2>&1 &

To resume training from the last saved checkpoint, append the `--resume` flag:

    nohup python -u train_student.py --resume > training.log 2>&1 &

*Note on Hardware:* If running on a 6GB VRAM GPU, ensure `train_student.py` is set to `batch_size=2`, `num_workers=1`, and `ACCUMULATION_STEPS=8` to prevent Out-of-Memory (OOM) crashes and system deadlocks.

### 3. Monitoring Background Logs
When executing training scripts in the background using `nohup`, you can monitor the real-time terminal output using `tail`:

    tail -f training.log

### 4. Run Inference & Visualization
Evaluate the models on held-out validation frames. Generates a side-by-side auto-scaled heatmap comparing predictions to ground truth.

    python predict_bev.py

*Outputs:* `debug_input_frame.jpg` and `prediction_output.jpg`

### 5. TensorBoard Integration
Launch TensorBoard to view Training/Validation metrics and real-time BEV grid output images.

    tensorboard --logdir=runs --bind_all