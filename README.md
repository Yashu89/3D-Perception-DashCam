# Robotic Camera Perception

A modular computer vision pipeline for **3D road-scene perception** using object detection, multi-object tracking, monocular depth estimation, 3D localization, velocity estimation, Time-to-Collision (TTC), and collision-risk classification.

The current prototype is tested on recorded driving videos and is structured for future real-time camera and robotics integration.

## Pipeline

```text
Video / Camera
      ↓
YOLO11 Object Detection
      ↓
ByteTrack Tracking
      ↓
Object Trajectory
      ↓
Depth Anything V2
      ↓
Depth Smoothing
      ↓
3D Position
      ↓
3D Velocity
      ↓
Closing Speed + TTC
      ↓
Collision Path
      ↓
Risk Classification
```

## Features

- YOLO11n object detection
- ByteTrack multi-object tracking
- Per-object trajectory history
- Monocular depth estimation using Depth Anything V2
- Temporal depth smoothing
- Approximate 3D object localization
- 3D velocity estimation
- Exponential Moving Average velocity smoothing
- Closing-speed estimation
- Time-to-Collision (TTC)
- Collision-path estimation
- SAFE / WARNING / HIGH risk classification
- Three-frame confirmation for risk-state changes
- Recorded-video processing
- Designed for future real-time camera use

## 3D Position Estimation

The project estimates approximate 3D coordinates using the pinhole-camera model:

```text
X = (cx - cx_camera) × Z / fx
Y = (cy - cy_camera) × Z / fy
Z = depth
```

The current implementation uses approximate camera parameters:

```text
fx = 700
fy = 700
cx_camera = 384
cy_camera = 216
```

These are **temporary approximations**, not calibrated camera intrinsics. The image center is currently used as the approximate camera principal point.

Errors in `fx`, `fy`, `cx`, and `cy` can introduce systematic errors in 3D position and consequently affect velocity, closing speed, TTC, and risk estimation.

For real-camera deployment, proper camera calibration is required.

## Depth and Performance

Depth estimation is performed at `640 × 360` to reduce computational cost.

On the development RTX 3050 system, depth inference reached approximately `18–22 ms/frame` at this resolution under the tested configuration.

Performance depends on input resolution, GPU/CPU, memory, model size, video properties, and the number of detected objects.

Higher resolution can improve visual detail and detection of small objects, but increases computational cost and latency. Lower resolution improves processing speed but can reduce detection, tracking, and depth quality.

The complete pipeline may require significant computational resources. Hardware with limited CPU/GPU capability may not maintain the same real-time performance.

## Current Limitations

The current implementation is a working perception prototype with several known engineering limitations:

- Camera intrinsics are currently approximate and should be calibrated for real-camera deployment.
- Monocular depth introduces uncertainty, especially under occlusion, reflections, and difficult weather or lighting conditions.
- Ego-motion compensation has not yet been integrated.
- Collision-path geometry is currently simplified.
- Processing performance depends strongly on hardware and input resolution.
- Risk thresholds are prototype engineering parameters and are not safety-certified.

These limitations define the next development stages rather than preventing the current perception pipeline from being demonstrated.

## Technical Notes

### Camera Parameters

The current 3D projection uses approximate values for `fx`, `fy`, `cx`, and `cy`. Because these parameters directly affect the conversion from image coordinates and depth to 3D coordinates, inaccuracies can propagate into velocity, closing-speed, and TTC estimation.

For real-camera deployment, these parameters should be obtained through camera calibration.

### Resolution vs Performance

The depth model currently operates at `640 × 360` to maintain practical processing speed.

Increasing resolution can improve visual detail and small-object detection, but also increases computational cost and latency. Lower resolution improves efficiency but may reduce detection, tracking, and depth quality.

Therefore, final performance depends on the trade-off between perception quality, resolution, latency, and available hardware.

### Hardware Requirements

The complete pipeline runs multiple computationally intensive components together:

```text
YOLO + ByteTrack + Depth Estimation + Image Processing
```

Systems with limited CPU/GPU resources may not maintain the same processing rate. Model optimization and asynchronous processing can be explored for lower-power or embedded platforms.

## Current Status

| Component | Status |
|---|---|
| YOLO detection | Complete |
| ByteTrack tracking | Complete |
| Trajectory estimation | Complete |
| Monocular depth | Complete |
| Depth smoothing | Complete |
| 3D position | Complete |
| 3D velocity | Complete |
| Velocity smoothing | Complete |
| Closing speed | Complete |
| TTC estimation | Complete |
| Collision-path estimation | Complete |
| Risk classification | Complete |
| 3-frame risk confirmation | Complete |
| Recorded-video testing | Complete |
| Camera calibration | Future |
| Ego-motion compensation | Future |
| Free-space / obstacle representation | Future |
| Navigation interface | Future |
| Real-time camera testing | Next stage |

## Repository Structure

```text
robotic_cam_perception/
├── inputs/
│   ├── detect.py
│   └── process.py
├── videos/
│   ├── driving_1.mp4
│   ├── driving_2.mp4
│   ├── driving_3.mp4
│   └── driving_4.mp4
├── outputs/
│   ├── driving_1_result.mp4
│   ├── driving_2_result.mp4
│   ├── driving_3_result.mp4
│   └── driving_4_result.mp4
├── README.md
├── requirements.txt
└── .gitignore
```

## Running

Create and activate the virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the required YOLO11n weights separately.

Process all videos:

```bash
python inputs/process.py
```

The generated results are saved in `outputs/`.

## Test Data

The current demonstration uses recorded driving footage for development and testing. The footage contains visible Getty Images watermarks and is not original camera data.

Before redistributing the original footage publicly, its applicable licensing and redistribution permissions should be verified.

## Technologies

- Python
- OpenCV
- YOLO11 / Ultralytics
- ByteTrack
- PyTorch
- Hugging Face Transformers
- Depth Anything V2
- NumPy
- CUDA

## Project Evolution

The current system establishes the core perception pipeline:

```text
Detection
    ↓
Tracking
    ↓
Depth
    ↓
3D Localization
    ↓
Motion Estimation
    ↓
TTC
    ↓
Risk Estimation
```

The planned progression is:

```text
Current Prototype
      ↓
Camera Calibration
      ↓
Real-Time Camera Testing
      ↓
Ego-Motion Compensation
      ↓
Metric 3D Validation
      ↓
Free-Space / Obstacle Representation
      ↓
ROS 2 / Robotics Integration
      ↓
Navigation and Planning Interface
```

The goal is to evolve the prototype from recorded-video perception into a validated real-time perception module that can provide structured 3D environmental information to downstream robotics systems.
