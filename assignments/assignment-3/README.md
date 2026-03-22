# Assignment 3 — Drone detection, Kalman tracking, and deliverables

Course submission code and documentation for this assignment live in **`assignments/assignment-3/`**. Scripts assume you run them from this directory (or adjust paths); large assets (Zenodo imagery, `videos/`, `detections/`, trained weights) are not vendored here—prepare data with `prepare_yolo_dataset.py` and place weights as needed.

This work builds on a **Zenodo UAV** image dataset prepared in YOLO layout (`prepare_yolo_dataset.py`), trains and runs **Ultralytics RT-DETR** for single-class **drone** detection, exports per-frame results, tracks the drone with a **filterpy** Kalman filter, and packages outputs for Hugging Face and YouTube.

---

## Hugging Face dataset (detections, Parquet)

**Dataset:** [rushilara/detections-final](https://huggingface.co/datasets/rushilara/detections-final/tree/main)

The repository contains **`data/detections.parquet`**: one row per frame with columns **`video`**, **`frame`**, and nested **`detections`** (class, confidence, bbox), exported from the per-frame JSON pipeline with PyArrow. Parquet is the canonical tabular format for this submission; it consolidates all frames into a single file suitable for `datasets` / analytics.

---

## Output tracking videos (YouTube)

Tracking videos (one per test input) were uploaded to a personal YouTube channel. GitHub does not allow iframes in README files; use the thumbnails below to open each video.

| Video | Link |
|--------|------|
| Tracking run 1 | [![Watch on YouTube](https://img.youtube.com/vi/_6Snk2gyysY/hqdefault.jpg)](https://www.youtube.com/watch?v=_6Snk2gyysY) |
| Tracking run 2 | [![Watch on YouTube](https://img.youtube.com/vi/yRjH0vPJB-4/hqdefault.jpg)](https://www.youtube.com/watch?v=yRjH0vPJB-4) |

Direct links:

- [https://www.youtube.com/watch?v=_6Snk2gyysY](https://www.youtube.com/watch?v=_6Snk2gyysY)
- [https://www.youtube.com/watch?v=yRjH0vPJB-4](https://www.youtube.com/watch?v=yRjH0vPJB-4)

---

## Dataset choice and detector configuration

**Training / evaluation data** comes from the **Zenodo UAV** dataset, arranged for Ultralytics as `data/yolo_ultralytics/` with `train` / `val` / `test` splits and a single class **`drone`** (`data/yolo_ultralytics/drone.yaml`, `nc: 1`, `names: {0: drone}`).

**Detector:** [Ultralytics](https://github.com/ultralytics/ultralytics) **RT-DETR** (Real-Time DEtection TRansformer), trained with `train_rtdetr.py`. Default configuration uses pretrained **`rtdetr-l.pt`**, image size **640**, batch size and workers tuned for GPU memory, **AdamW** optimizer, cosine LR schedule, and early stopping on validation. Inference on video is performed with `run_video_inference.py`, which runs RT-DETR on each frame and writes JSON detections (`class`, `confidence`, `bbox`) for downstream tracking.

---

## Kalman filter: state design and noise parameters

Tracking is implemented in `kalman_track_video.py` using **filterpy**’s `KalmanFilter`.

**State vector (4D):** constant-velocity model in the image plane  

\[
\mathbf{x} = [c_x,\, c_y,\, v_{x},\, v_{y}]^\top
\]

where \((c_x, c_y)\) is the **center of the detector bounding box** in pixels and \((v_x, v_y)\) is velocity in pixels per frame. **Measurements** are \((c_x, c_y)\) from the chosen detection each frame (`H` observes only position).

**Motion model:** discrete-time \(F\) with \(\Delta t = 1\) frame, standard constant-velocity structure.

**Noise / covariance (as implemented):**

| Symbol | Role | Value |
|--------|------|--------|
| **P** | Initial state covariance (scaled after default init) | diagonal scaled by **500** |
| **R** | Measurement noise (2×2, position) | **20 · I** |
| **Q** | Process noise (4×4) | diagonal **0.05** on position states, **0.5** on velocity states (indices 2–3) |

At track creation, the state is initialized from the first detection center with **zero velocity**. The highest-**confidence** `"drone"` detection is used when multiple boxes appear (`best_drone_detection`).

---

## Failure cases and missed detections

**Missed detections (no box or empty JSON for a frame):** the filter still runs **`predict()`** every frame. If there is **no** measurement, **`update()`** is skipped: the track **coasts** using the predicted center and velocity. A counter records **consecutive** missed frames; if it exceeds **`--max-missed`** (default **8**), the track is **dropped** (state cleared, trajectory reset). If the drone appears again later, a **new** track starts at the next detection. This limits how long the overlay follows prediction alone and avoids indefinite drift when the detector fails for an extended period.

**Other failure modes:**

- **No detection to start:** frames are skipped until the first positive detection initializes the filter.
- **Multiple detections:** a single global track uses the **highest-confidence** drone box, which can jump if the wrong box wins—mitigated by the filter smoothing and by the miss threshold when the true target disappears.
- **Video decode:** AV1 or problematic codecs may require **`--decoder ffmpeg`** so frames align with detection indices.

Output videos include only frames where the track is **active** (after first detection until loss or end), with **detector box** (when present), **Kalman center**, and **trajectory polyline** over estimated centers.

---

## Repository map (main scripts)

| Script | Purpose |
|--------|---------|
| `prepare_yolo_dataset.py` | Zenodo UAV → Ultralytics YOLO layout |
| `train_rtdetr.py` | Fine-tune RT-DETR |
| `run_video_inference.py` | Per-frame RT-DETR JSON |
| `kalman_track_video.py` | filterpy Kalman tracking + overlay videos |
| `export_detections_parquet.py` / `upload_detections_to_hf.py` | Parquet export and Hugging Face upload |

---

## Citation (tools / models)

- **Ultralytics RT-DETR** — Jocher, Chaurasia, Qiu, et al.; Ultralytics YOLO repository: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **filterpy** — Roger Labbe: [https://github.com/rlabbe/filterpy](https://github.com/rlabbe/filterpy)
- **Zenodo UAV dataset** — Rafael Makrigiorgis, Nicolas Souli, & Panayiotis Kolios. (2022). *Unmanned Aerial Vehicles Dataset* (1.0) [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.7477569](https://doi.org/10.5281/zenodo.7477569)
