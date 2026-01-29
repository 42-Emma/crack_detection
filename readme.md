rack Detection Node
A ROS2 node for real-time crack detection in structural inspection using deep learning models (UNet and Pix2Pix). This node processes camera images to detect and localize cracks, providing comprehensive detection results with temporal filtering, depth-based validation, and visualization capabilities.
Features

Dual Model Support: Switch between UNet and Pix2Pix architectures
Real-time Detection: Processes camera streams with configurable frame skipping
Temporal Filtering: Reduces false positives through multi-frame consistency checks
Depth Integration: Optional depth-based filtering to validate detections at specific distances
Zoom Capability: Configurable digital zoom for detailed inspection
Scan Point Extraction: Identifies crack center and extent (start/end points)
Robot Pose Tracking: Associates detections with AMCL robot poses
Visualization: Multiple visualization outputs including detection masks, scan points, and depth overlays
Capture System: Trigger-based image capture with metadata for later analysis

Installation
Prerequisites

ROS2 (tested on Humble/Iron)
Python 3.8+
PyTorch
OpenCV
CUDA (optional, for GPU acceleration)

Required ROS2 Dependencies
bashsudo apt install ros-<distro>-cv-bridge ros-<distro>-sensor-msgs
Python Dependencies
bashpip install torch torchvision opencv-python numpy
```

## 🛠 Prerequisites & Weights

To keep the repository lightweight, the large trained weight files are not included in the Git history.

### Download the Weights

1. **U-Net Weights** (`unet_weights.pth`)
2. **Pix2Pix Weights** (`pix2pix_weights.pth`)

### Placement

Ensure the `.pth` files are placed in the `models/` directory within the package root:
```
crack_detection/models/
├── unet_weights.pth
└── pix2pix_weights.pth
Configuration
The node is highly configurable through ROS2 parameters. Key parameters include:
Model Configuration

model_type: Model architecture ('unet' or 'pix2pix')
unet_model_path: Path to UNet weights
pix2pix_model_path: Path to Pix2Pix weights

Detection Parameters

min_crack_percent: Minimum crack coverage percentage for detection (default: 0.5%)
confidence_threshold: Binary classification threshold (default: 0.5)
skip_frames: Process every Nth frame (default: 1)

Temporal Filtering

temporal_window_size: Number of frames for consistency check (default: 5)
temporal_threshold: Required detection ratio (default: 0.6)

Hysteresis Filtering

hysteresis_upper_threshold: Upper threshold for crack percentage
hysteresis_lower_threshold: Lower threshold for crack percentage

Depth Filtering

depth_filtering_enabled: Enable/disable depth-based filtering
depth_min_distance: Minimum valid depth distance (m)
depth_max_distance: Maximum valid depth distance (m)

Zoom Configuration

zoom_enabled: Enable digital zoom
zoom_factor: Zoom magnification factor
zoom_center_x, zoom_center_y: Zoom region center

Visualization

publish_visualization: Enable visualization publishing
save_detections: Enable automatic saving of detections

Topics
Subscribed Topics
TopicTypeDescription/camera/image_rawsensor_msgs/ImageInput camera stream/camera/depth/image_rawsensor_msgs/ImageDepth image (optional)/amcl_posegeometry_msgs/PoseStampedRobot pose from AMCL/capture_triggerstd_msgs/BoolTrigger to save current detection
Published Topics
TopicTypeDescription/crack_detection/resultstd_msgs/StringJSON detection result/crack_detection/detectedstd_msgs/BoolBinary detection flag/crack_detection/crack_centergeometry_msgs/PointCrack center in image coordinates/crack_detection/robot_posegeometry_msgs/PoseStampedRobot pose at detection/crack_detection/visualizationsensor_msgs/ImageVisualization overlay/crack_detection/scan_visualizationsensor_msgs/ImageScan points visualization/crack_detection/markervisualization_msgs/MarkerRViz marker for 3D visualization
