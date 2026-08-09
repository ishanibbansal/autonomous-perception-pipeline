### Log 3.7: The Camera Lift-Splat Upgrade and Explicit Depth Supervision
* **Date:** August 8, 2026
* **Symptom:** The single-frame camera student suffered from severe geometric distortion because `AdaptiveAvgPool2d` destroyed the perspective-to-BEV spatial mapping. 
* **Solution:** Completely replaced the naive pooling block with a `LiftSplatViewTransformer`. The ResNet-50 backbone now outputs a discrete depth distribution (48 bins, 2m-50m) alongside contextual features. Used Waymo camera intrinsics to mathematically project the 2D pixels into a 3D frustum, and pooled them into the 160x160 BEV grid.
* **Secondary Upgrade (BEVDepth Style):** Because the batch inherently contains the Teacher's fused LiDAR point clouds, we geometrically projected the LiDAR onto the camera's image plane to generate sparse, ground-truth depth maps. Added a `CrossEntropyLoss` depth branch to explicitly supervise the camera's `depth_net`, anchoring its spatial predictions in physical reality.

### Log 3.8: Hardware Reality Check: Single-Frame Distillation vs Temporal Fusion
* **Date:** August 8, 2026
* **Symptom:** Initial integration of a $T=3$ temporal loop and 3D Convolution fusion block caused batch times to skyrocket. Training projections leaped from 16 hours to over 70+ hours, risking severe VRAM OOM crashes on the 6GB RTX 2060.
* **Strategic Pivot:** Stripped out the explicit temporal loop from the Student model. The architecture now relies entirely on Cross-Modal Distillation (forcing the Student to mimic the temporal Teacher's intermediate features) and single-frame Lift-Splat spatial geometry. Batch times successfully collapsed to ~0.7 seconds/batch (completing 15 epochs in ~7.5 hours).

### Log 3.9: PyTorch Broadcasting and Resolution Mismatch Bugs
* **Date:** August 8, 2026
* **Symptom:** The Student forward pass crashed during `LiftSplat` execution with two distinct errors:
  1. `RuntimeError: view size is not compatible with input tensor's size and stride`
  2. `IndexError: The shape of the mask [115200] at index 0 does not match the shape of the indexed tensor [460800, 64]`
* **Root Cause:** 
  1. `.permute()` operations fragmented memory contiguity, causing subsequent `.view()` operations to crash.
  2. The network's spatial grids were hardcoded to a $640 \times 960$ baseline assumption, creating a $40 \times 60$ 3D frustum. However, the actual Waymo input images were $1280 \times 1920$, causing the ResNet `layer3` to output an $80 \times 120$ feature map. The sizes violently clashed during `scatter_add_` voxel pooling.
* **Solution:** 
  1. Replaced all `.view()` calls during voxel pooling with `.reshape()`, which safely handles non-contiguous permuted tensors.
  2. Stripped all $640 \times 960$ hardcodes. Expanded the frustum to exactly match the native $80 \times 120$ feature map output. Updated the dataset horizontal flip augmentations to invert across $1920.0$ pixels instead of $960.0$.