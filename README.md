# Waymo BEV Perception Pipeline: Cross-Modal Distillation & Edge Deployment

A 2D/3D autonomous vehicle perception pipeline trained on the Waymo Open Dataset. This repository implements a **Teacher-Student Knowledge Distillation** architecture to transfer forward-facing spatial geometry (a 70x80 meter physical grid) from a heavy, multi-modal sensor fusion model (LiDAR + Camera) into a lightweight, vision-only monocular model (Camera only), followed by a **C++ TensorRT & ROS 2** edge deployment pipeline.

## Features

* **Multi-Modal Teacher Network:** Fuses historical LiDAR sweeps (`num_sweeps=3`) using a PointPillars encoder and a ResNet perspective camera backbone via a BiFPN Bird's-Eye View (BEV) decoder.
* **Monocular Student Network:** A lightweight, camera-only architecture trained to mimic the Teacher's intermediate semantic feature maps and dense occupancy grids within a 70m deep by 80m wide forward-facing spatial grid (`x_range=(0.0, 70.0)`, `y_range=(-40.0, 40.0)`).
* **CenterNet Spatial Encoding:** Replaces rigid binary grids with continuous 2D Gaussian heatmaps (160x160 resolution) for stable gradient optimization. A strict forward-FOV filter (`X > 2.0`) automatically excludes non-visible rear geometry.
* **TensorRT Acceleration & C++ ROS 2 Node:** Decouples inference from Python by compiling frozen static ONNX graphs into a TensorRT execution engine (`student_bev.engine`). Wrapped in an asynchronous ROS 2 C++ node (`student_bev_node`) for real-time edge processing and side-by-side visualization publishing.
* **Hardware-Optimized Training:** Designed to train dual deep neural networks simultaneously on hardware constrained to 6GB-8GB of VRAM (e.g., NVIDIA RTX 2060) utilizing strict gradient accumulation and optimized multiprocessing.

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
    │       ├── train/               # Waymo .tfrecord training files
    │       └── val/                 # Waymo .tfrecord validation files
    ├── docs/                        # Documentation and engineering logs
    │   └── dev_log.md
    ├── scripts/                     # Pipeline utility and training scripts
    │   ├── export_onnx.py           # Static ONNX graph exporter
    │   ├── test_ros_node.py         # Test publisher for ROS 2 deployment
    │   ├── train_student.py         # Phase 1: Spatial Initialization
    │   ├── train_student_temporal.py # Phase 2: Temporal Fusion
    │   ├── train_student_e2e.py     # Phase 3: End-to-End Fine-Tuning
    │   └── predict_student_temporal.py # Temporal visualization tool
    ├── src/                         # Core architecture and ROS 2 workspace
    │   └── perception/
    │       ├── models/              # Teacher, Student, and Loss architectures
    │       ├── utils/               # Dataset parsing and spatial metrics
    │       └── ros_nodes/
    │           └── student_bev_inference/
    │               ├── CMakeLists.txt
    │               ├── package.xml
    │               └── src/
    │                   └── student_bev_node.cpp
    ├── .gitignore                   # Git ignore file
    └── README.md                    # Project documentation

---

## Curriculum Learning Pipeline

This repository utilizes a Curriculum Learning approach, splitting the distillation process into three distinct phases to prevent catastrophic forgetting and strictly manage GPU VRAM limits.

*   **Phase 1: Spatial Initialization**
    *   Trains the `LiftSplatViewTransformer` and spatial backbone on single frames.
    *   Establishes the core 2D-to-3D geometric mapping.
*   **Phase 2: Temporal Fusion**
    *   **Freezes** the spatial backbone.
    *   Introduces a T=3 chronological sequence loop (t-2, t-1, t0).
    *   Trains the temporal fusion block to align past BEV features using ego-motion extrinsics to track velocity and handle occlusions.
*   **Phase 3: End-to-End Fine-Tuning**
    *   **Unfreezes** the entire network architecture.
    *   Applies micro learning rates (`1e-5`) to allow temporal loss gradients to optimize the ResNet camera feature extractor, minimizing depth smearing ("comet tails").

---

## Installation

1. **Clone the repository:**

        git clone https://github.com/ishanibbansal/autonomous-perception-pipeline.git
        cd autonomous-perception-pipeline

2. **Install Python Dependencies:**
   Ensure you have PyTorch (with CUDA support) and TensorFlow (strictly for parsing `.tfrecord` files). 

        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install tensorflow opencv-python matplotlib tensorboard onnx

3. **Install ROS 2 & TensorRT Dependencies:**
   Ensure ROS 2 and TensorRT 11+ runtime headers and development libraries are installed.

---

## Usage

### 1. Train the Models
Execute the training scripts in order to build the fully featured temporal network. Use `nohup` for safe background execution.

* **Train the Multi-Modal Teacher:**
  ```bash
  nohup python -u scripts/train_teacher.py > teacher_training.log 2>&1 &
  ```

* **Phase 1: Spatial Initialization (Student):**
  ```bash
  nohup python -u scripts/train_student.py > phase1_training.log 2>&1 &
  ```

* **Phase 2: Temporal Fusion (Student):**
  ```bash
  nohup python -u scripts/train_student_temporal.py > phase2_training.log 2>&1 &
  ```

* **Phase 3: End-to-End Fine-Tuning (Student):**
  ```bash
  nohup python -u scripts/train_student_e2e.py > phase3_training.log 2>&1 &
  ```

### 2. Export and Build TensorRT Engine

* **Export static graph to ONNX:**
  ```bash
  python3 scripts/export_onnx.py
  ```

* **Compile to TensorRT Engine:**
  ```bash
  trtexec --onnx=student_bev.onnx --saveEngine=student_bev.engine
  ```

### 3. Build & Run ROS 2 Inference Node

* **Build the package:**
  ```bash
  colcon build --packages-select student_bev_inference
  source install/setup.bash
  ```

* **Run the inference node:**
  ```bash
  ros2 run student_bev_inference bev_node --ros-args -p engine_path:=student_bev.engine
  ```