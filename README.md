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
*Documentation coming soon...*

## Model Training (Sprint 2)
*Documentation coming soon...*

## Telemetry & Visualization (Sprint 3)
*Documentation coming soon...*