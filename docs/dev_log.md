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

## 🧠 Perception Architecture & Data Pipeline Logs (Sprint 1)

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

## 👁️ Perception Training & Inference Logs (Sprint 2)

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