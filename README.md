# Waymo BEV Perception Pipeline: Distillation and Edge Deployment

[![CI - Geometry & Math Verification](https://github.com/ishanibbansal/autonomous-perception-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/ishanibbansal/autonomous-perception-pipeline/actions/workflows/ci.yml)

A perception stack trained on the Waymo Open Dataset. This repository covers sensor ingestion, teacher-student distillation, BEV detection, evaluation, and a C++ TensorRT ROS 2 inference node. It does not include a tracker or a motion planner.

A **Teacher-Student Knowledge Distillation** setup transfers 3D geometric perception from a multi-modal teacher (LiDAR + camera) into a lightweight vision-only monocular student. A Sub-Pixel NMS decoder turns the student's occupancy heatmap into physical `[x, y, length, width, heading]` boxes.

## Features

* **Multi-Modal Teacher Network:** Fuses historical LiDAR sweeps (`num_sweeps=3`) using a PointPillars encoder and a ResNet perspective camera backbone via a BiFPN Bird's-Eye View (BEV) decoder.
* **Monocular Student Network (Curriculum Learning):** A lightweight, camera-only architecture trained to mimic the Teacher's intermediate semantic feature maps and dense occupancy grids within a 70m deep by 80m wide forward-facing spatial grid.
* **Mathematical BEV-to-Physical Bridge:** A Sub-Pixel NMS decoder turns continuous probability heatmaps and tensor plateaus into rigid `[x, y, length, width, heading]` boxes.
* **C++ TensorRT ROS 2 Node:** `student_bev_inference` (`bev_node`) subscribes to `/camera/image_raw` and publishes a BEV occupancy heatmap on `/perception/bev_heatmap`. Training and evaluation stay in Python.
* **Hardware-Optimized Training:** Designed for 6GB-8GB of VRAM (e.g., NVIDIA RTX 2060) using gradient accumulation and detached temporal feature caches.

---

## Edge Deployment (C++ & TensorRT)

Python/PyTorch inference is too slow for a real-time perception budget, so the deployed path is a C++ TensorRT node.

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

```bash
export PATH=$PATH:/usr/lib/wsl/lib
```

---

## Directory Structure

```text
├── .github/workflows/ci.yml       # CPU geometry and distillation-loss tests
├── data/raw/{train,val}/         # Waymo .tfrecord files (gitignored, not shipped)
├── docs/dev_log.md               # Engineering scratchpad, not a user guide
├── scripts/
│   ├── benchmarks/
│   │   ├── eval_3d/             # 3D / BEV / center-distance evaluation
│   │   └── latency_trt/         # C++ TensorRT latency profiler
│   ├── train_student.py         # Phase 1: spatial initialization
│   ├── train_student_temporal.py
│   ├── train_student_e2e.py     # Phase 3: end-to-end fine-tuning
│   ├── export_onnx.py
│   ├── compare_pipelines.py     # PyTorch vs. C++ TensorRT parity
│   └── test_ros_node.py
├── src/perception/
│   ├── models/                  # Teacher and student
│   ├── losses/
│   ├── utils/                   # NMS decoder, target encoder, dataset
│   └── ros_nodes/student_bev_inference/  # C++ TensorRT ROS 2 node
├── tests/                       # CPU geometry and distillation-loss tests
├── requirements-ci.txt          # pytest / numpy / scipy for those tests
└── requirements.txt            # Frozen ROS 2 machine dump; do not pip install
```

---

## The Perception Training Curriculum

This repository utilizes a Curriculum Learning approach, splitting the distillation process into three distinct phases to prevent catastrophic forgetting and strictly manage GPU VRAM limits.

*   **Phase 1: Spatial Initialization:** Trains the `LiftSplatViewTransformer` and spatial backbone on single frames to establish the core 2D-to-3D geometric mapping.
*   **Phase 2: Temporal Fusion:** Freezes the spatial backbone and introduces a T=3 chronological sequence loop (t-2, t-1, t0) to track velocity and handle occlusions.
*   **Phase 3: End-to-End Fine-Tuning:** Unfreezes the entire network architecture with micro learning rates (`1e-5`) to allow temporal loss gradients to optimize the ResNet camera feature extractor.

---

## Benchmarks & Performance

Numbers below are logged results, not CI checks. Geometry unit tests in `tests/` run on CPU and do not require Waymo records, checkpoints, ROS 2, or a GPU. Full-val detection and TensorRT latency were measured locally; see `docs/dev_log.md` (Log 3.20 and Log 3.22) for the evaluation protocol.

### Detection Accuracy (Waymo Open Dataset Val Split)

Strict camera-only monocular detection from `best_e2e_student_checkpoint.pt` (Log 3.22). Winning protocol: `--no-flip-y`, `se3_inverse(extrinsics)`, `--score-thresh 0.05`, `--min-radius 1.2`.

Center-distance is nuScenes-style Euclidean matching of object centers. It is not volumetric IoU, and it does not use predicted box extents. The same run's 3D AP@0.5 is 0.95% because the dimension head does not produce usable extents.

| Metric | Result |
| --- | --- |
| Overall Center-Distance AP (CD@4.0m) | 30.62% |
| Vehicle Center-Distance AP (CD@4.0m) | 36.16% |
| Overall 3D AP@0.5 | 0.95% |
| Vehicle 3D AP@0.5 | 0.95% |
| Evaluated Frames | 991 frames (15,393 ground-truth objects) |
| Evaluation split / notes | Strict camera-only monocular detection |

### Edge Hardware Latency (RTX 2060)

C++ TensorRT latency profiler in `scripts/benchmarks/latency_trt/`, recorded in Log 3.20 on an NVIDIA GeForce RTX 2060.

| Metric | Result |
| --- | --- |
| Engine / Precision | TensorRT 11 (FP16) |
| Mean Latency | 54.08 ms |
| Throughput | 18.5 FPS |
| P95 Latency | 54.32 ms |
| VRAM Budget | < 6.0 GB VRAM allocation limit |

TensorRT `.engine` plans are hardware-architecture specific and compiled on-device. See docs/dev_log.md for compilation profiles.

---

## Installation

### Geometry tests (no dataset, no GPU, no ROS)

This is the only path that runs from a fresh clone.

```bash
git clone https://github.com/ishanibbansal/autonomous-perception-pipeline.git
cd autonomous-perception-pipeline
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-ci.txt
PYTHONPATH=. pytest tests/test_canonical_frame.py tests/test_distill_loss.py -v
```

`torchvision` is required because `src/perception/models/student_bev.py` imports it at module load. The tests never download weights or run the network.

Do not `pip install -r requirements.txt`. That file is a freeze of a ROS 2 Humble machine, including `rclpy`, TensorFlow 2.11, and CUDA 11.8 wheels. It will not install in a normal virtualenv.

### Training

Training and full-val evaluation need the Waymo Open Dataset under `data/raw/{train,val}` and a CUDA GPU. Checkpoints and `.tfrecord` files are gitignored.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow opencv-python matplotlib tensorboard onnx
```

TensorFlow is only used to parse `.tfrecord` files.

### ROS 2 and TensorRT

The C++ node is `src/perception/ros_nodes/student_bev_inference`. It needs ROS 2 Humble (Ubuntu 22.04) and TensorRT 11 development libraries. `student_bev.engine` is compiled on the target GPU and is not in the repository; an engine built for another GPU will not load.
