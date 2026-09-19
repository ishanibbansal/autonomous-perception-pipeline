# Waymo BEV Perception Pipeline: Distillation, Edge Deployment, and Motion Planning

A complete autonomous vehicle R&D stack trained on the Waymo Open Dataset. This repository implements a full pipeline from raw sensor ingestion to trajectory generation. 

It features a **Teacher-Student Knowledge Distillation** architecture transferring 3D geometric perception from a heavy multi-modal sensor fusion model (LiDAR + Camera) into a lightweight vision-only monocular model. The perception outputs are bridged via rigorous Non-Maximum Suppression (NMS) math into a distributed **ROS 2 Prediction and Planning** stack to track objects and generate collision-free trajectories.

## Features

* **Multi-Modal Teacher Network:** Fuses historical LiDAR sweeps (`num_sweeps=3`) using a PointPillars encoder and a ResNet perspective camera backbone via a BiFPN Bird's-Eye View (BEV) decoder.
* **Monocular Student Network (Curriculum Learning):** A lightweight, camera-only architecture trained to mimic the Teacher's intermediate semantic feature maps and dense occupancy grids within a 70m deep by 80m wide forward-facing spatial grid.
* **Mathematical BEV-to-Physical Bridge:** Implements a custom Sub-Pixel NMS Decoder to translate continuous probability heatmaps and tensor plateaus into rigid, physical `[x, y, length, width, heading]` bounding boxes for robotic planners.
* **Distributed ROS 2 Architecture (`rclpy` & `rclcpp`):** Decouples the machine learning pipeline from the robotics pipeline. Uses Python ROS 2 nodes for rapid trajectory and Kalman Filter R&D, while supporting a C++ TensorRT edge deployment pipeline for the monocular student (`student_bev_node`).
* **Hardware-Optimized Training:** Designed to train dual deep neural networks simultaneously on hardware constrained to 6GB-8GB of VRAM (e.g., NVIDIA RTX 2060) utilizing strict gradient accumulation and optimized multiprocessing.

---

## Edge Deployment (C++ & TensorRT)

The primary engineering constraint of this pipeline is real-time autonomous execution. Relying on Python/PyTorch for inference introduces unacceptable overhead for vehicle motion planning.

To bridge this, the Student monocular model is deployed via a bare-metal C++ TensorRT ROS 2 package (`student_bev_inference`), validated to achieve full numerical and spatial parity with the PyTorch baseline:
* **Memory Management:** Replaces Python's dynamic garbage collection with pre-allocated contiguous GPU/CPU memory pools for input and output tensors.
* **Precision Targeting:** Compiles the ONNX graph into a TensorRT engine to maximize throughput on NVIDIA hardware while preserving 3D occupancy geometry.
* **ROS 2 Integration:** Wrapped in a zero-copy `rclcpp` Node (`bev_node`) subscribing to `/camera/image_raw` and publishing 3D BEV occupancy heatmaps to `/perception/bev_heatmap`.

**Building & Running the C++ Node:**
```bash
# Build the ROS 2 package
colcon build --packages-select student_bev_inference

# Terminal 1: Launch the C++ inference node
source install/setup.bash
ros2 run student_bev_inference bev_node --ros-args -p engine_path:=student_bev.engine

# Terminal 2: Publish a test image and evaluate output
python3 scripts/test_ros_node.py
```

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
    │   ├── compare_pipelines.py     # PyTorch vs. C++ TensorRT parity verification tool
    │   ├── test_ros_node.py         # ROS 2 test publisher and heatmap evaluator
    │   ├── extract_calib.py         # Camera intrinsic/extrinsic calibration extractor
    │   ├── export_onnx.py           # Static ONNX graph exporter
    │   ├── train_student.py         # Phase 1: Spatial Initialization
    │   ├── train_student_temporal.py # Phase 2: Temporal Fusion
    │   ├── train_student_e2e.py     # Phase 3: End-to-End Fine-Tuning
    │   ├── predict_student.py       # Monocular Student BEV prediction visualizer
    │   └── predict_bev.py           # NMS Bounding Box visualizer
    ├── src/                         # Core architecture and ROS 2 workspace
    │   ├── perception/
    │   │   ├── models/              # Teacher, Student, and Loss architectures
    │   │   └── utils/               # NMS Decoders, target encoding, and datasets
    │   ├── prediction/              # [NEW] EKF Object Tracking
    │   ├── planning/                # [NEW] Trajectory Generation (MPC / Pure Pursuit)
    │   └── ros_nodes/               # ROS 2 Deployment Workspace
    │       ├── student_bev_inference/ # C++ TensorRT Edge Node
    │       └── perception_bridge/     # Python rclpy Teacher Output Node
    ├── .gitignore                   # Git ignore file
    └── README.md                    # Project documentation

---

## The Perception Training Curriculum

This repository utilizes a Curriculum Learning approach, splitting the distillation process into three distinct phases to prevent catastrophic forgetting and strictly manage GPU VRAM limits.

*   **Phase 1: Spatial Initialization:** Trains the `LiftSplatViewTransformer` and spatial backbone on single frames to establish the core 2D-to-3D geometric mapping.
*   **Phase 2: Temporal Fusion:** Freezes the spatial backbone and introduces a T=3 chronological sequence loop (t-2, t-1, t0) to track velocity and handle occlusions.
*   **Phase 3: End-to-End Fine-Tuning:** Unfreezes the entire network architecture with micro learning rates (`1e-5`) to allow temporal loss gradients to optimize the ResNet camera feature extractor.

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
