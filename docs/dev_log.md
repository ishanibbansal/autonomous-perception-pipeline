# Engineering Log & Scratchpad

This file serves as a living technical journal for the project. It tracks active debugging sessions, environment quirks, and architectural decisions. Capturing these roadblocks and solutions builds a concrete knowledge base for technical deep-dives and portfolio reviews.

---

## 🛠️ Environment & Infrastructure Logs (Sprint 0)

### Log 0.1: Remote SSH Terminal Missing `nvidia-smi` Path
* **Date:** July 17, 2026
* **Symptom:** Running `nvidia-smi` locally in WSL2 works perfectly, but executing it over a remote VS Code SSH session throws: `bash: nvidia-smi: command not found`.
* **Root Cause:** WSL2 automatically injects Windows host paths (including the folder where the virtualized NVIDIA drivers live) during standard local logins. However, remote non-interactive or standard SSH sessions skip this injection, leaving the shell blind to `/usr/lib/wsl/lib`.
* **Solution:** Manually append the WSL driver directory to the system path inside the user configuration:

      echo 'export PATH=$PATH:/usr/lib/wsl/lib' >> ~/.bashrc
      source ~/.bashrc

---

## 🧠 Perception Architecture & Data Pipeline Logs

### Log 1.1: Expanding Target Labels for Monocular 3D Vision
* **Date:** July 23, 2026
* **Symptom:** The standard dataloader was insufficient for training the perception pipeline to understand physical depth and volume. 
* **Root Cause:** Standard object detection datasets default to extracting 2D bounding boxes `[x_min, y_min, x_max, y_max]`, leaving the loss function blind to physical world metrics.
* **Solution:** Updated the machine learning dataset class to explicitly capture the 3D bounding box target labels. Configured the parser to extract the `[X, Y, Z, Length, Width, Height, Heading]` metrics directly from the Waymo `.tfrecord` laser annotations to properly train the spatial regression model.

### Log 1.2: The "Pixels vs. Meters" Grid Collision
* **Date:** July 23, 2026
* **Symptom:** The training loop achieved highly accurate Validation Loss (e.g., `0.0143`), but the Validation mAP score strictly stalled at `0.0000` after 50 epochs. Model outputs were extremely small fractions (e.g., predicting a car length of `0.74` instead of `4.8` meters).
* **Root Cause:** `TargetEncoder` was taking 3D real-world coordinates (meters) and dividing them by the 2D image pixel stride. This mathematically crushed every single vehicle into the top-left `(0, 0)` cell of the 40x60 spatial grid. The model learned to predict perfectly encoded sub-grid fractions instead of physical meters, meaning the physical volumetric IoU overlap requirement was mathematically impossible to meet.
* **Solution:** Split the extraction logic into a 10D tensor: `[Class, Pix_X, Pix_Y, 3D_X, 3D_Y, 3D_Z, L, W, H, Heading]`. Used the 2D pixel coordinates to map the vehicles onto the 40x60 grid, and assigned the raw un-normalized 3D meters directly as the regression targets for the loss function.

### Log 1.3: The 360-Degree LiDAR Bug & Sensor Fusion
* **Date:** July 23, 2026
* **Symptom:** Attempting to filter visible vehicles by cross-referencing Waymo's `camera_labels` IDs with `laser_labels` IDs resulted in `0` intersecting boxes.
* **Root Cause:** 1) Waymo assigns completely separate, non-matching string IDs for human-annotated 2D images and 3D LiDAR (for vehicles). 2) LiDAR scans a 360-degree radius. Iterating purely through `laser_labels` was feeding the neural network 3D coordinates for vehicles located *behind* the ego-vehicle's front camera.
* **Solution:** Bypassed the dataset's human tracking IDs and implemented mathematical sensor fusion. Utilized a pinhole camera model (`focal_length = 2000.0`, `camera_height = 1.5m`) to geometrically project the physical 3D LiDAR coordinates directly onto the 2D image plane `(u, v)`. Applied a strict forward-FOV filter (`X > 2.0`) to immediately exclude non-visible geometry.

---

## 👁️ Perception Training & Inference Logs

### Log 2.1: Training Plateau and Duplicate Bounding Box Clustering
* **Date:** July 25, 2026
* **Symptom:** After completing a training run using the frozen YOLOv8 backbone and custom `Head3D`, validation mAP remained critically low and validation loss plateaued around `30 to 37` while training loss steadily descended.
* **Root Cause:** 1) Missing or un-tuned Non-Max Suppression (NMS) during post-processing, allowing multiple adjacent grid cells and anchor priors to independently fire positive predictions for the same object. 2) The "Frozen Backbone Wall": YOLOv8's backbone was pre-trained entirely on 2D COCO objects, which contain zero metric depth, orientation, or 3D bounding box information. A completely frozen feature extractor made it mathematically impossible for the custom head to extract true spatial depth features.
* **Solution:** Completed task `#17 Build the Inference Pipeline` (`predict.py`) and implemented NMS clustering. Unfroze the tail of the YOLOv8 backbone (layers 5–9) while keeping early feature layers (0–4) frozen, and introduced differential learning rates (`1e-5` for the backbone, `1e-3` for the `Head3D`) to allow proper geometric adaptation without destroying pre-trained edge weights. Added gradient norm clipping (`max_norm=5.0`) and best-checkpoint tracking to secure peak validation performance.

---

## 🚀 Architectural Redesign: CenterNet & Bin-Based Depth (July 2026)

### Log 2.2: Breaking the Monocular 2% mAP Ceiling via Architectural Overhaul
* **Date:** July 26, 2026
* **Symptom:** Baseline continuous monocular regression plateaued at ~2% mAP with near-zero confidence conviction. The network struggled because strict spatial boundaries and erratic depth loss gradients caused the objectness head to output near-zero probabilities everywhere.
* **Root Cause:** 1) Forcing hard, binary single-cell assignments created massive spatial friction when features were slightly shifted. 2) Strict continuous depth regression treated distance errors equally at all ranges, breaking human-like perception logic (where close-range errors are critical and far-range errors are less significant).
* **Solution:** Performed a full architectural redesign of `target_encoder.py`, `model.py`, `loss.py`, and `validate.py`:
  1. **CenterNet Gaussian Heatmaps:** Replaced strict binary grid cells with continuous 2D Gaussian splats (`sigma=1.0`) and applied Penalty-Reduced Focal Loss. Used a 3x3 max-pooling peak extraction method during inference to isolate local maxima.
  2. **Bin-Based Depth Classification + Residual:** Mapped continuous forward depth into discrete bins (e.g., 40 bins up to 80 meters) trained with Cross-Entropy loss, combined with a sigmoid-constrained local residual branch (`SmoothL1Loss`) for fine-grained depth precision.
  3. **Stabilized Gradients:** Allowed the network to organically learn spatial and metric distributions instead of harsh, all-or-nothing grid constraints, driving immediate validation improvements right out of Epoch 1.
  4. **Stable Orientation & Dimension Regression:** Utilized continuous sine/cosine angle representations ($\sin(\text{yaw}), \cos(\text{yaw})$) instead of raw radians to prevent angular discontinuity wraparound bugs during loss calculation, paired with direct metric regression for length, width, and height.

### Log 2.3: Reaching 10% Monocular mAP & The Transition to Modular BEV Fusion
* **Date:** July 26, 2026
* **Milestone:** Reached a peak Validation mAP of **0.1017 (10.17%)** at Epoch 38 on an IoU threshold of 0.25 using pure monocular 3D detection with CenterNet heatmaps, discrete bin-based depth classification, and target-aware data augmentations (`ColorJitter` + horizontal spatial flipping).
* **Symptom & Bottleneck:** 
  1. While Training Loss dropped steadily from `37.62` down to `0.8437`, Validation Loss diverged upward to `12.9696`.
  2. The massive spread between training and validation loss indicated clear overfitting: the deep feature extractor memorized the lighting, background structures, and vehicle shadows of the ~2,000 training frames.
  3. Inferring 3D spatial depth purely from 2D pixel scale remains an ill-posed monocular problem. Low-confidence false positives in the background inflated the grid focal loss, while primary vehicle bounding boxes suffered from physical depth jitter on unseen validation scenes.
* **Strategic Architectural Pivot:** Transitioned from single-view 2D image-plane prediction to a **Modular Bird's-Eye View (BEV) Sensor Fusion Architecture**, combining perspective camera semantics with physical LiDAR point cloud telemetry.
* **Codebase Integration & Reusability:**
  * **YOLOv8 Backbone (`model.py`):** Retained 100% to extract high-level 2D semantic feature representations from perspective front-camera frames.
  * **Data Pipeline (`WaymoDataset` in `dataset.py`):** Maintained all existing photometric augmentations, horizontal flipping logic (`y = -y`, `heading = -heading`), and camera intrinsic matrices ($F_x, F_y, C_x, C_y$). Expanded the parser to read raw point cloud tensors directly from the `.tfrecord` laser returns to extract exact physical depth measurements.
  * **Loss & Head Modules (`loss.py`):** Retained the CenterNet Focal Loss for objectness classification along with Smooth L1 / Cross-Entropy loss for depth bins, dimensions, and orientations. Re-projected the target coordinate space from image-plane pixels $(u, v)$ to top-down physical ego-vehicle grid cells $(X, Y)$ on a metric 2D BEV plane.
  * **Downstream Alignment:** Transforming predictions onto a top-down BEV occupancy grid establishes the 1:1 metric spatial mapping mandatory for downstream trajectory prediction (calculating $\frac{dX}{dt}, \frac{dY}{dt}$) and motion planning.

### Log 2.4: The Mirrored BEV Coordinate Grid Bug
* **Date:** July 29, 2026
* **Symptom:** TensorBoard visualizer revealed that the predicted BEV heatmaps were perfectly horizontally mirrored compared to the ground truth grid. The model correctly identified vehicle locations and shapes, but projected them onto the opposite side of the road.
* **Root Cause:** In `target_encoder.py`, the target calculation explicitly inverted the lateral axis to map world Y to camera X (`self.grid_w - ...`). However, the CNN feature extractor naturally preserves the left-to-right pixel mapping of the camera all the way through the spatial projection. Flipping the ground truth targets but not the features forced the model to learn a horizontally mirrored projection.
* **Solution:** Removed the explicit inversion (`self.grid_w -`) from the grid corner calculation inside `target_encoder.py`. Restarted the training run with a clean TensorBoard cache, instantly resolving the axis collision and aligning the prediction gradients perfectly with the ground truth targets.

### Log 2.5: Combating BEV Overfitting with Aggressive Regularization
* **Date:** July 29, 2026
* **Symptom:** During Epoch 6 of the BEV fusion architecture, training loss aggressively dropped to `0.5400`, creating highly accurate visualizations on the training frames. However, Validation Soft-IoU was cut in half (dropping from a peak of `0.0178` down to `0.0088`), indicating textbook data memorization.
* **Root Cause:** The BEV head was assigned an aggressive learning rate of `1e-3` from Epoch 1 with weak weight decay. The network quickly forged brittle, high-magnitude edge weights to map specific static backgrounds and artifacts of the `.tfrecord` training splits rather than learning invariant geometry. 
* **Solution:** Paused training and injected three regularization safeguards:
  1. **Weight Decay Increase:** Bumped `weight_decay` in the AdamW optimizer from `1e-4` to `1e-3` to penalize overly confident, brittle feature dependencies.
  2. **Linear LR Warmup:** Introduced `SequentialLR` to apply a 2-epoch linear warmup (starting at `0.1x` of the base rate) before shifting into the cosine decay schedule. This prevents the head from locking into poor spatial representations early in training.
  3. **Targeted Perturbations:** Amplified `ColorJitter` ranges and introduced randomized sharpness scaling to the `WaymoDataset` pipeline. This actively distorts static photometric cues, forcing the backbone to encode true structural outlines of the vehicles instead of local color artifacts.

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

### Log 3.16: Bridging Perception to Planning (The NMS Peak Extractor)
* **Date:** August 21, 2026
* **Objective:** Translate the Teacher model's continuous probability heatmaps into rigid physical coordinates (X, Y, length, width, heading) required for downstream robotic trajectory planning.
* **Symptom:** Direct $3 \times 3$ max-pooling spawned dozens of overlapping green bounding boxes for a single vehicle. Box orientations were also mirrored relative to the Ground Truth.
* **Root Cause:** 
  1. **The Flat Plateau Bug:** The neural network's sigmoid activations saturated at exactly `1.000` over large clusters of pixels. `nonzero()` evaluated all adjacent pixels as the "maximum", spawning 5-10 duplicate boxes per vehicle.
  2. **Top-K Starvation:** Attempting to fix the plateaus using `torch.topk(30)` mathematically truncated genuine detections. A single saturated car plateau would consume 15 of the 30 available slots, causing the network to completely ignore other cars in the scene.
  3. **Mirrored Math:** The target grid was horizontally flipped (`torch.flip(..., dims=[-1])`) to align visual space, but the neural network's sub-pixel offset (`offset_x`) and orientation vector (`sin_h`) predictions were not negated, causing boxes to point in reverse.
* **Solution:** 
  1. Implemented an industrial-grade Non-Maximum Suppression (NMS) decoder using a larger $5 \times 5$ max-pooling kernel.
  2. Reverted to full `nonzero()` extraction to prevent starvation, and applied an aggressive 2.5-meter Euclidean radius suppression sweep to guarantee single-box-per-vehicle outputs.
  3. Negated `offset_x` and `sin_h` in the decoding loop to mathematically align the vectors with the flipped coordinate space.
* **Next Steps (Phase Transition):** The Perception pipeline is now structurally complete. Small amounts of hallucinated false positives/negatives remain in the output, which is standard for raw sensor inference. The project is officially transitioning to the **Prediction and Planning** phase. The next step is building an Extended Kalman Filter (EKF) tracking node in ROS 2 (`rclpy`) to filter out ghost tracks and enforce object permanence across time.

### Log 3.17: C++ TensorRT vs. Python Inference Parity & Coordinate Alignment Fix
* **Date:** September 19, 2026
* **Branch:** `feature/cpp-inference-parity`
* **Symptoms:** 
  1. The C++ TensorRT node produced significantly worse confidence / empty heatmaps compared to the Python `predict_student.py` model on test images.
  2. Detected vehicle positions in the BEV appeared flipped relative to the camera image (e.g., closest vehicles on the left appeared farthest, and farthest vehicles on the right appeared closest).
* **Root Causes:**
  1. **Image Degradation & Calibration Drift:** `test_frame.jpg` was an old, lossy 640×960 JPEG upscaled 2× to 1280×1920, starving the ResNet50 backbone of high-frequency spatial features. Additionally, the C++ node used mismatched/inverted extrinsics.
  2. **Double Inversion Bug (Revisiting Log 3.12):** In `BEVGridEncoder` and `LiftSplatViewTransformer`, grid row indexing is defined as $\text{row} = 159 - \frac{X - X_{\min}}{\text{res}_x}$. Row 0 represents $X = 70\text{ m}$ (far ahead) while row 159 represents $X = 0\text{ m}$ (ego bumper). In standard matrix/image coordinate conventions (OpenCV), row 0 is at the top and row 159 is at the bottom, so the raw output was *already* canonically oriented for top-down driving. Matplotlib's `origin='lower'` in `predict_student.py` inverted this display, leading to the erroneous `cv::flip(heatmap, heatmap, 0)` introduced in Log 3.12.
* **Solutions:**
  1. Authored `scripts/compare_pipelines.py` to extract a pristine, lossless 1280×1920 PNG (`test_frame_clean.png`) and dump exact per-frame raw extrinsics and intrinsics. Updated `test_ros_node.py` to publish the clean frame.
  2. Updated `student_bev_node.cpp` with matching calibration and added runtime min/max probability logging. TensorRT peak confidence jumped from near zero to **0.7746**.
  3. Removed `cv::flip(heatmap, heatmap, 0)` from `student_bev_node.cpp`. Corrected plotting to `origin='upper'` across `compare_pipelines.py`, `predict_student.py`, and `predict_student_temporal.py`.
* **Outcome:** The C++ TensorRT ROS 2 inference node achieved complete numerical and spatial parity with the PyTorch model, correctly placing near-left vehicles at bottom-left and receding vehicles at top-right.