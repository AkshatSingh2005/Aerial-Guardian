# Aerial Guardian: Drone-Based Multi-Person Detection and Tracking

## Overview

Aerial Guardian is a lightweight UAV perception system designed for detecting and tracking multiple people in drone footage. The project uses YOLOv8n for person detection and ByteTrack for multi-object tracking on the VisDrone MOT dataset.

To address the challenge of small object detection in aerial imagery, a tiled inference enhancement was introduced. This approach divides each frame into smaller regions before detection, improving the visibility of distant pedestrians and increasing detection performance while maintaining near real-time operation.

---

## Problem Statement

Detecting and tracking people from a drone presents several challenges:

* People occupy very few pixels in high-altitude footage.
* Camera motion introduces tracking instability.
* Occlusions can cause identity switches.
* Real-time performance is required for deployment on edge devices.

The goal of this project is to develop a lightweight system capable of detecting, tracking, and visualizing multiple persons in UAV footage while maintaining real-time performance.

---

## Dataset

### VisDrone2019-MOT Validation Set

The VisDrone dataset contains drone-captured video sequences collected in various urban environments.

Characteristics:

* Moving UAV camera
* Small-scale pedestrians
* Occlusions
* Perspective distortion
* Crowded scenes

Target Class:

* Person

---

## System Architecture

### Baseline Pipeline

Input Video Frames
→ YOLOv8n Person Detection
→ ByteTrack Multi-Object Tracking
→ Trajectory Visualization
→ Output Video

### Proposed Enhanced Pipeline

Input Video Frames
→ Tiled Frame Generation (2×2)
→ YOLOv8n Detection per Tile
→ Non-Maximum Suppression (NMS)
→ ByteTrack Tracking
→ Trajectory Visualization
→ Output Video

---

## Technologies Used

### Detection

* YOLOv8n
* Ultralytics

### Tracking

* ByteTrack
* Supervision Library

### Computer Vision

* OpenCV

### Language

* Python 3

---

## Key Features

### Person Detection

YOLOv8n is used as a lightweight detector capable of real-time inference while maintaining reasonable detection accuracy.

### Multi-Object Tracking

ByteTrack assigns persistent IDs to detected persons and maintains tracks across frames.

### Trajectory Visualization

Historical positions of tracked individuals are visualized using short trajectory tails.

### Small Object Enhancement

A tiled inference strategy was introduced to improve detection of distant pedestrians commonly found in drone imagery.

---

## Experimental Results

### Baseline Detection

* Persons Detected: 20

### Tiled Detection + NMS

* Persons Detected: 37

### Improvement

* Detection Increase: ~85%

### Tracking Performance

* Frames Processed: 464
* Processing Time: 12.26 seconds
* Average FPS: 37.83

The system operates above real-time requirements while significantly improving small-object detection.

---

# Assignment Discussion & Engineering Decisions

## 1. Architecture Choice

The system follows a modular perception pipeline consisting of:

**Input Frames → YOLOv8n Detector → ByteTrack Tracker → Trajectory Visualization**

YOLOv8n was selected because the assignment emphasized lightweight deployment and real-time performance. Compared to larger YOLO variants, YOLOv8n provides a strong balance between accuracy, speed, and model size (~6 MB).

ByteTrack was chosen as the tracking algorithm because it is computationally efficient and performs robust data association using both high-confidence and low-confidence detections, making it suitable for UAV surveillance scenarios.

The final system achieved approximately **37.83 FPS** on the VisDrone sequence while maintaining stable multi-person tracking.

---

## 2. Handling Small Object Detection

One of the major challenges in drone imagery is that pedestrians occupy only a small number of pixels due to the high camera altitude. In the baseline configuration, YOLOv8n detected **20 persons** in the test frame.

To address this issue, a **2×2 tiled inference strategy** was introduced.

Instead of running detection on the entire frame, the frame is divided into four smaller tiles. YOLOv8n performs inference on each tile independently, after which the detections are merged and duplicate bounding boxes are removed using Non-Maximum Suppression (NMS).

This approach effectively increases the relative size of pedestrians within the detector's input space and improves small-object visibility.

### Results

| Method              | Persons Detected |
| ------------------- | ---------------- |
| Baseline YOLOv8n    | 20               |
| Tiled YOLOv8n       | 39               |
| Tiled YOLOv8n + NMS | 37               |

This represents an approximately **85% increase in detected persons** compared to the baseline detector.

---

## 3. Handling ID Switching, Drone Motion, and Occlusions

Drone footage introduces frequent camera motion, partial occlusions, and temporary detection failures. These effects can lead to ID switching, where the same person receives a new tracking ID after disappearing for several frames.

To mitigate this problem, ByteTrack was used as the tracking backend because it associates detections using both high-confidence and low-confidence observations rather than discarding weaker detections immediately.

Additionally, trajectory history was maintained and visualized for each track, allowing short-term motion continuity even when detections fluctuate.

Although full ego-motion compensation was not implemented in this lightweight version, the system maintains reasonable track consistency for moderate camera movement. Future improvements could include:

* Camera ego-motion estimation using feature matching and homography estimation.
* Appearance-based Re-Identification (ReID).
* StrongSORT or DeepSORT integration.
* Kalman filter parameter optimization.

---

## 4. Edge Deployment Strategy (NVIDIA Jetson)

The project was designed with edge deployment in mind.

Several decisions support deployment on embedded hardware:

* YOLOv8n was selected instead of larger models to reduce memory usage and inference latency.
* ByteTrack introduces minimal computational overhead.
* The complete detector model size remains well below the 300 MB requirement.
* The pipeline maintains real-time performance while remaining lightweight.

For deployment on NVIDIA Jetson platforms, the following optimizations would be applied:

1. Export YOLOv8n to ONNX format.
2. Optimize inference using TensorRT.
3. Use FP16 precision to reduce memory usage and improve throughput.
4. Dynamically enable tiled inference only when small-object density is high.
5. Adjust image resolution depending on available compute resources.

These optimizations would allow the system to maintain strong detection performance while operating within edge-device constraints.

---

## 5. Engineering Trade-Offs

This project intentionally balances accuracy and computational efficiency.

The baseline detector offers higher speed but misses many distant pedestrians. The tiled inference enhancement significantly improves detection performance but requires additional inference passes per frame.

Therefore, the key trade-off is:

**Higher Detection Accuracy ↔ Increased Computation**

For surveillance and safety-critical UAV applications, the improved detection rate can justify the additional computation cost.

The final system demonstrates that meaningful gains in small-object detection can be achieved through lightweight engineering modifications without requiring larger models or expensive retraining.

---

## Future Work

* Camera ego-motion compensation
* Track-guided detection
* Appearance-based re-identification
* Adaptive tile sizing
* Dynamic confidence thresholds
* Edge deployment benchmarking on Jetson hardware

---

## Conclusion

This project demonstrates a lightweight UAV surveillance pipeline capable of real-time person detection and tracking. Beyond the baseline YOLOv8n + ByteTrack architecture, a tiled inference enhancement was introduced to improve small-object detection in aerial imagery, increasing detections from 20 to 37 instances while maintaining real-time performance.

The resulting system provides an effective balance between detection accuracy, tracking quality, and computational efficiency, making it suitable for deployment on resource-constrained aerial platforms.
