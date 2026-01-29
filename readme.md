# Crack Detection Node

A ROS2 node for real-time crack detection in structural inspection using deep learning models (UNet and Pix2Pix). This node processes camera images to detect and localize cracks, providing comprehensive detection results with temporal filtering, depth-based validation, and visualization capabilities.

## Features

- **Dual Model Support**: Switch between UNet and Pix2Pix architectures
- **Real-time Detection**: Processes camera streams with configurable frame skipping
- **Temporal Filtering**: Reduces false positives through multi-frame consistency checks
- **Depth Integration**: Optional depth-based filtering to validate detections at specific distances
- **Zoom Capability**: Configurable digital zoom for detailed inspection
- **Scan Point Extraction**: Identifies crack center and extent (start/end points)
- **Robot Pose Tracking**: Associates detections with AMCL robot poses
- **Visualization**: Multiple visualization outputs including detection masks, scan points, and depth overlays
- **Capture System**: Trigger-based image capture with metadata for later analysis

## Installation

### Prerequisites

- ROS2 (tested on Humble/Iron)
- Python 3.8+
- PyTorch
- OpenCV
- CUDA (optional, for GPU acceleration)

### Required ROS2 Dependencies

```bash
sudo apt install ros-<distro>-cv-bridge ros-<distro>-sensor-msgs
```

### Python Dependencies

```bash
pip install torch torchvision opencv-python numpy
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
```

## Configuration

The node is highly configurable through ROS2 parameters. Key parameters include:

### Model Configuration

- `model_type`: Model architecture (`'unet'` or `'pix2pix'`)
- `unet_model_path`: Path to UNet weights
- `pix2pix_model_path`: Path to Pix2Pix weights

### Detection Parameters

- `min_crack_percent`: Minimum crack coverage percentage for detection (default: 0.5%)
- `confidence_threshold`: Binary classification threshold (default: 0.5)
- `skip_frames`: Process every Nth frame (default: 1)

### Temporal Filtering

- `temporal_window_size`: Number of frames for consistency check (default: 5)
- `temporal_threshold`: Required detection ratio (default: 0.6)

### Hysteresis Filtering

- `hysteresis_upper_threshold`: Upper threshold for crack percentage
- `hysteresis_lower_threshold`: Lower threshold for crack percentage

### Depth Filtering

- `depth_filtering_enabled`: Enable/disable depth-based filtering
- `depth_min_distance`: Minimum valid depth distance (m)
- `depth_max_distance`: Maximum valid depth distance (m)

### Zoom Configuration

- `zoom_enabled`: Enable digital zoom
- `zoom_factor`: Zoom magnification factor
- `zoom_center_x`, `zoom_center_y`: Zoom region center

### Visualization

- `publish_visualization`: Enable visualization publishing
- `save_detections`: Enable automatic saving of detections

## Topics

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Input camera stream |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | Depth image (optional) |
| `/amcl_pose` | `geometry_msgs/PoseStamped` | Robot pose from AMCL |
| `/capture_trigger` | `std_msgs/Bool` | Trigger to save current detection |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/crack_detection/result` | `std_msgs/String` | JSON detection result |
| `/crack_detection/detected` | `std_msgs/Bool` | Binary detection flag |
| `/crack_detection/crack_center` | `geometry_msgs/Point` | Crack center in image coordinates |
| `/crack_detection/robot_pose` | `geometry_msgs/PoseStamped` | Robot pose at detection |
| `/crack_detection/visualization` | `sensor_msgs/Image` | Visualization overlay |
| `/crack_detection/scan_visualization` | `sensor_msgs/Image` | Scan points visualization |
| `/crack_detection/marker` | `visualization_msgs/Marker` | RViz marker for 3D visualization |

## Usage

### Basic Launch

```bash
ros2 launch crack_detection crack_detection.launch.py
```

### Launch with Custom Parameters

You can override parameters in your launch file or via command line:

```bash
ros2 launch crack_detection crack_detection.launch.py model_type:=pix2pix min_crack_percent:=1.0
```

### Triggering Image Capture

Publish to the capture trigger topic:

```bash
ros2 topic pub /capture_trigger std_msgs/Bool "data: true" --once
```

## Output Data

### Detection Result Format

The `/crack_detection/result` topic publishes JSON-formatted detection data:

```json
{
  "detected": true,
  "crack_percentage": 2.35,
  "center_x": 320,
  "center_y": 240,
  "frame_count": 1523,
  "temporal_status": "5/5 frames",
  "wall_distance": 1.25,
  "in_detection_state": true,
  "timestamp_sec": 1234567890,
  "timestamp_nanosec": 123456789
}
```

### Saved Captures

When triggered, the node saves:
- Original image
- Binary detection mask
- Probability heatmap
- Depth image (if available)
- Metadata JSON with detection details and scan points

Files are saved to `~/crack_detection_captures/` with timestamps.

## License

MIT

---

**Developed for structural inspection and autonomous crack detection in ROS2 environments.**
