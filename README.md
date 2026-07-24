# Autonomous Vehicle Perception Pipeline

A 2D/3D perception pipeline trained on the Waymo Open Dataset, featuring a distributed development architecture. Model training and data processing are executed on a headless Windows/WSL2 GPU host, while real-time metrics and telemetry are routed to a macOS client via a Tailscale mesh network.

## System Architecture

This project utilizes a distributed compute model to decouple the development UI from the hardware execution:
* **Host (Compute Engine):** Windows PC running Ubuntu via WSL2 (NVIDIA RTX 2060).
* **Client (Control Center):** MacBook Pro running VS Code Remote-SSH and Foxglove Studio.
* **Network:** Zero-configuration secure tunnel via Tailscale (`100.x.x.x` subnet).

## Environment Setup (Sprint 0)

To replicate this environment, both the host and client must be authenticated on the same Tailscale network. 

1. **Host Configuration:**
   - Install WSL2 and Ubuntu.
   - Install `openssh-server` and `tailscale`.
   - Export WSL GPU drivers to the Linux path: `export PATH=$PATH:/usr/lib/wsl/lib`
   
2. **Client Configuration:**
   - Add the host's Tailscale IP to `~/.ssh/config`.
   - Connect via VS Code Remote-SSH.

## Data Pipeline (Sprint 1)

The pipeline ingests raw `.tfrecord` files from the Waymo Open Dataset and prepares them for monocular 3D spatial regression. 

* **Monocular 3D Labels:** The custom machine learning dataset class explicitly captures 3D bounding box target labels (`[X, Y, Z, Length, Width, Height, Heading]`) to properly train the network on physical depth and volume metrics.
* **Sensor Fusion & Geometric Projection:** To reconcile mismatched human-annotated IDs across sensors, the pipeline uses a pinhole camera model (`focal_length = 2000.0`, `camera_height = 1.5m`) to mathematically project 3D LiDAR geometry onto the 2D front camera plane. A forward-FOV filter dynamically drops any labels physically located behind the ego-vehicle.
* **Spatial Grid Encoding:** Bounding boxes are processed into a 10D tensor. The 2D image pixels are used to map vehicles to a discrete 40x60 spatial grid, while the raw, un-normalized 3D meters are preserved and assigned to the grid cells as physical regression targets for the loss function.

## Model Training (Sprint 2)
*Documentation coming soon...*

## Telemetry & Visualization (Sprint 3)
*Documentation coming soon...*