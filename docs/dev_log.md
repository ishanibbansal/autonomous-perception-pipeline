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

### Log 3.10: Geometric Contradictions and The Mirrored Teacher
* **Date:** August 12, 2026
* **Symptom:** The distillation model was completely stuck at a 0.0510 validation IoU, then crashed into a "Confidence Valley" of 0.0001, performing significantly worse than a pure monocular baseline.
* **Root Cause:** Two massive spatial alignment failures shattered the coordinate space:
  1. **The Upside-Down Projector:** The Student's Lift-Splat-Shoot `voxel_pooling` mapped +X to the bottom of the grid and +Y to the right, physically projecting vehicles upside-down and backward relative to the `TargetEncoder` (which maps +X to Top and +Y to Left).
  2. **The Mirrored Teacher:** The Teacher's PointPillar logic inverted the X-axis but skipped the Y-axis inversion, resulting in horizontally mirrored feature maps.
* **Solution:** Mathematically inverted the X and Y indices (`159 - x_idx`, `159 - y_idx`) within the Student's `voxel_pooling` logic to perfectly match the target grid origin. Restored a horizontal `torch.flip(..., dims=[-1])` to the Teacher's intermediate feature maps in both `train_student.py` and `validate.py` to synchronize the distillation coordinate space.

### Log 3.11: The "Telepathy" Penalty and Loss Bottlenecks
* **Date:** August 12, 2026
* **Symptom:** Even after fixing the geometry, the validation score crept up at a microscopic rate (0.0002 per epoch), and the detection Dice loss was severely plateaued near 0.99.
* **Root Cause:** 
  1. **The Telepathy Penalty:** The distillation `dataset.py` lacked the physical camera FOV filter. It was mapping *all* 360-degree LiDAR vehicles onto the target grid, violently punishing the monocular Student for failing to detect invisible cars in its physical blind spots.
  2. **Loss Bullying:** The auxiliary depth loss (`alpha_depth=1.0`) was swinging wildly, completely drowning out the tiny focal gradients required to actually predict bounding boxes.
  3. **Zero-Safe Strategy:** The Student lacked a background prior bias, causing it to take massive early focal penalties and default to predicting 0 everywhere.
* **Solution:** 
  1. Restored the strict physical FOV filter (`x > 2.0 and abs(y / x) < 0.6`) in `dataset.py` and implemented a spatial FOV mask in `distill_loss.py` to strictly evaluate the Student inside the physical camera cone.
  2. Slashed `alpha_depth` to `0.1` to un-choke the bounding box detection gradients.
  3. Added a `prior_prob = 0.01` bias initialization to the final occupancy head to establish a 99% empty road baseline before training starts.

### Log 3.12: Edge Deployment: ONNX Export, TensorRT Engine Acceleration & ROS 2 C++ Node Integration
* **Date:** August 15, 2026
* **Objective:** Transition the trained PyTorch Monocular Student network into a production-grade, real-time C++ inference pipeline using TensorRT and ROS 2.
* **Challenges & Solutions:**
  1. **Dynamic Shapes & Plugin Deserialization in TensorRT 11:** 
     * *Symptom:* Initial ONNX conversions failed during engine deserialization in C++ with `API Usage Error (Cannot find plugin: ScatterElements, version: 2)`.
     * *Root Cause:* Dynamic shape tracing introduced runtime graph uncertainty, and TensorRT 11 encapsulates `ScatterElements` inside `libnvinfer_plugin.so.11`.
     * *Fix:* Wrapped the model forward pass in `BEVExportWrapper` with fixed Waymo projection matrices to freeze static tensor shapes (`1x3x1280x1920` image, `1x3x3` intrinsics, `1x4x4` extrinsics). Linked `/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.11` in CMake and forward-declared `initLibNvInferPlugins()` to initialize plugin registries prior to engine deserialization.
  2. **Preprocessing Distribution Mismatch:** 
     * *Symptom:* Initial C++ inference output produced a uniform, high-confidence magenta block of false positives across the entire BEV grid.
     * *Root Cause:* The C++ pipeline applied standard ImageNet normalization (mean/std subtraction), whereas the PyTorch training pipeline consumed raw `[0.0, 1.0]` RGB floats.
     * *Fix:* Removed ImageNet normalization from `student_bev_node.cpp`, feeding pristine `[0, 1]` floats directly into CUDA device buffers.
  3. **Spatial Coordinate System Alignment:** 
     * *Symptom:* Predicted occupancy clusters appeared inverted along the vertical axis.
     * *Root Cause:* OpenCV renders matrices with origin at the top-left, whereas PyTorch BEV target grids anchor $(X=0, Y=0)$ at the bottom.
     * *Fix:* Applied vertical inversion (`cv::flip(heatmap, heatmap, 0)`) before colormap mapping and generated a side-by-side stitched frame (`cv::hconcat`) combining the camera perspective with the 3D BEV occupancy heatmap.

### Log 3.13: The WSL Swap Thrashing Wall and Dataloader Memory Management
* **Date:** August 18, 2026
* **Symptom:** Training batch times sporadically spiked from 0.5 s/batch to 1.2–3.3 s/batch, specifically at epoch boundaries or after prolonged continuous uptime. 
* **Root Cause:** A hard collision between PyTorch's multiprocessing and physical RAM limits. With 32GB of total system RAM, Windows background processes (12GB), WSL allocations, and PyTorch's `num_workers=8` triggered the memory "Cliff Effect." PyTorch's Copy-on-Write (CoW) memory duplication maxed out the remaining RAM, forcing the Linux kernel to aggressively push the WSL virtual machine into the NVMe swap file, crippling CPU execution speeds.
* **Solution:** 
  1. Reduced `num_workers` to `4` for training and `2` for validation to keep the dataset footprint safely within physical RAM.
  2. Set `persistent_workers=False` to enforce strict garbage collection and destroy the memory bloat at the end of every epoch. The pipeline locked back into a stable 0.5 s/batch.

### Log 3.14: Phase 2 Temporal Dataset Dimension Mismatch & Alignment
* **Date:** August 18, 2026
* **Symptom:** Phase 2 initialization instantly crashed with `RuntimeError: mat1 and mat2 shapes cannot be multiplied (52945x70 and 71x64)` in the Teacher's PointPillar encoder. A subsequent run failed with `NameError: name 'F' is not defined` inside `student_bev.py`.
* **Root Cause:** 
  1. Transitioning from `WaymoDataset` to `WaymoTemporalDataset` with `num_sweeps=0` bypassed the logic that appended the `dt_sec` (time-delta) column to the LiDAR point cloud, leaving the array at 70 channels instead of the Teacher's strictly expected 71.
  2. The temporal fusion `align_past_features` function was never executed during Phase 1 spatial training, masking a missing `torch.nn.functional` import used for the `affine_grid` transformation.
* **Solution:** 
  1. Manually appended a dummy `dt_sec = 0.0` feature column to the $t_0$ LiDAR points inside `WaymoTemporalDataset` to satisfy the Teacher's expected input shape.
  2. Added the missing `import torch.nn.functional as F` to the student model architecture file.

### Log 3.15: Temporal Efficacy and The "Comet Tail" Effect
* **Date:** August 18, 2026
* **Objective:** Validate the implementation of the temporal history loop ($t_{-2} \rightarrow t_0$) and assess its impact on Birds-Eye-View perception compared to the spatial baseline.
* **Results:** The Phase 2 Temporal Student reached a validation score of **13.7% EMA**, successfully doubling the Phase 1 spatial baseline of 6.2%. 
* **Observations:** Authored a chronological visualizer (`predict_student_temporal.py`) to map predictions. The model accurately maps lane topology and object persistence through occlusions. As expected with monocular physics, predictions exhibited the "Comet Tail" effect—tight lateral localization with elongated vertical depth smearing due to inherent 2D-to-3D geometric uncertainty. 
* **Next Steps:** Proceeding to Phase 3 End-to-End (E2E) Fine-Tuning. The spatial backbone will be unfrozen with a micro learning rate (`1e-5`) to allow temporal gradients to optimize the ResNet camera extractor.