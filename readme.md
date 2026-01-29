# Crack Detection ROS2 Node

A sophisticated ROS2 package for real-time crack detection in structural inspection applications. This node uses deep learning models (UNet and Pix2Pix) to detect and localize cracks in camera images, with advanced features like depth filtering, temporal consistency, and AMCL pose integration.

## Features

- **Dual Model Support**: UNet for fast detection, optional Pix2Pix for enhanced refinement
- **Tiled Inference**: Process high-resolution images with overlapping tiles
- **Depth-Based Filtering**: Filter out false positives using depth camera data
- **Temporal Consistency**: Reduce flickering with multi-frame detection voting
- **Hysteresis Control**: Smooth detection transitions with configurable thresholds
- **Digital Zoom**: Focus on specific regions of interest with 2D zoom
- **AMCL Integration**: Track and visualize robot pose during crack detection
- **RViz Markers**: Automatically place markers at crack locations on the map
- **Image Capture**: Save original, prediction, and mask images on demand
- **Performance Optimization**: Frame skipping and GPU acceleration support

## System Requirements

### Hardware
- NVIDIA GPU (recommended for real-time performance)
- RGB-D Camera (e.g., Intel RealSense, compatible with ROS2)
- Robot platform with AMCL localization (optional)

### Software
- Ubuntu 22.04 (recommended)
- ROS2 Humble or later
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)

## Dependencies

### ROS2 Packages
```bash
sudo apt install ros-humble-cv-bridge
sudo apt install ros-humble-sensor-msgs
sudo apt install ros-humble-visualization-msgs
```

### Python Packages
```bash
pip install torch torchvision  # PyTorch with CUDA support
pip install opencv-python
pip install numpy
pip install albumentations
```

## Installation

1. **Create workspace and clone repository**
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone <your-repo-url> crack_detection
```

2. **Download model weights**
Place your trained models in the package:
```bash
mkdir -p ~/ros2_ws/src/crack_detection/models
# Copy your models:
# - best_model.pth (UNet model)
# - pix2pix_epoch_98_best.pth (optional Pix2Pix model)
```

3. **Build the package**
```bash
cd ~/ros2_ws
colcon build --packages-select crack_detection
source install/setup.bash
```

4. **Verify installation**
```bash
ros2 pkg list | grep crack_detection
```

## Package Structure

```
crack_detection/
├── config/
│   └── crack_detection_params.yaml    # Configuration file
├── launch/
│   └── crack_detection_launch.py      # Launch file
├── crack_detection/
│   ├── crack_detection_node.py        # Main node
│   ├── unet_model.py                  # UNet architecture
│   ├── train_pix2pix.py              # Pix2Pix generator
│   ├── unet_rt_inference_tiling6.py  # Tiled UNet inference
│   └── pix2pix_rt_inference_tiling.py # Tiled Pix2Pix inference
├── models/
│   ├── best_model.pth                 # UNet weights
│   └── pix2pix_epoch_98_best.pth     # Pix2Pix weights (optional)
└── README.md
```

## Usage

### Quick Start

**Basic launch (fast mode):**
```bash
ros2 launch crack_detection crack_detection_launch.py
```

**With Pix2Pix refinement:**
```bash
ros2 launch crack_detection crack_detection_launch.py use_pix2pix:=true
```

**With tiling (for high-resolution images):**
```bash
ros2 launch crack_detection crack_detection_launch.py use_tiling:=true
```

**Custom camera topic:**
```bash
ros2 launch crack_detection crack_detection_launch.py \
    camera_topic:=/your/camera/topic \
    depth_topic:=/your/depth/topic
```

### Advanced Configuration

**Adjust detection sensitivity:**
```bash
ros2 launch crack_detection crack_detection_launch.py \
    threshold:=0.6 \
    min_crack_percent:=3.0
```

**Enable zoom for detailed inspection:**
```bash
ros2 launch crack_detection crack_detection_launch.py \
    zoom_enabled:=true \
    zoom_factor:=2.5 \
    zoom_center_x:=0.5 \
    zoom_center_y:=0.5
```

**Adjust temporal filtering (for moving platforms):**
```bash
ros2 launch crack_detection crack_detection_launch.py \
    temporal_window_size:=7 \
    temporal_threshold:=5
```

**Configure depth filtering:**
```bash
ros2 launch crack_detection crack_detection_launch.py \
    depth_filtering_enabled:=true \
    use_adaptive_depth:=true \
    depth_tolerance:=0.15
```

## Configuration Parameters

### Model Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | `models/best_model.pth` | Path to UNet model weights |
| `pix2pix_model_path` | string | `models/pix2pix_epoch_98_best.pth` | Path to Pix2Pix model weights |
| `use_tiling` | bool | `false` | Enable tiled inference for large images |
| `use_pix2pix` | bool | `false` | Enable Pix2Pix refinement (slower, better quality) |

### Detection Settings
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | int | `384` | Input resolution (256, 384, or 512) |
| `threshold` | float | `0.5` | Binary classification threshold (0.0-1.0) |
| `min_crack_percent` | float | `2.0` | Minimum crack coverage to trigger detection (%) |
| `skip_frames` | int | `1` | Process every Nth frame (1=all frames) |

### Tiling Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | int | `384` | Tile size for overlapping inference |
| `subdivisions` | int | `2` | Overlap factor (2=50%, 4=75% overlap) |

### Zoom Settings
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `zoom_enabled` | bool | `false` | Enable digital zoom |
| `zoom_factor` | float | `2.0` | Zoom magnification (1.0=no zoom) |
| `zoom_center_x` | float | `0.5` | Horizontal center (0.0-1.0, 0.5=center) |
| `zoom_center_y` | float | `0.5` | Vertical center (0.0-1.0, 0.5=center) |

### Temporal Consistency
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temporal_window_size` | int | `5` | Number of frames to consider |
| `temporal_threshold` | int | `4` | Minimum frames that must detect (out of window_size) |

### Hysteresis Control
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hysteresis_enabled` | bool | `true` | Enable hysteresis to prevent flickering |
| `hysteresis_low_threshold` | float | `1.0` | Turn OFF below this % (lower than min_crack_percent) |
| `hysteresis_min_duration` | float | `0.5` | Minimum detection duration in seconds |

### Depth Filtering
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `depth_filtering_enabled` | bool | `true` | Enable depth-based filtering |
| `use_adaptive_depth` | bool | `true` | Auto-adapt to current wall distance |
| `target_inspection_distance` | float | `1.5` | Target wall distance in meters (if not adaptive) |
| `depth_tolerance` | float | `0.1` | Depth tolerance in meters (±tolerance) |

### Image Capture
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_directory` | string | `crack_captures` | Directory to save captured images |
| `capture_key_topic` | string | `/capture_image` | Topic to trigger image capture |

### ROS Topics
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `camera_topic` | string | `/a200_1103/sensors/camera_0/color/image` | Input camera topic |
| `depth_topic` | string | `/a200_1103/sensors/camera_0/depth/image` | Input depth topic |
| `publish_visualization` | bool | `true` | Publish visualization images |

## ROS Topics

### Subscribed Topics
| Topic | Type | Description |
|-------|------|-------------|
| `<camera_topic>` | `sensor_msgs/Image` | Input RGB camera stream |
| `<depth_topic>` | `sensor_msgs/Image` | Input depth camera stream |
| `/amcl_pose` | `geometry_msgs/PoseWithCovarianceStamped` | Robot localization for crack mapping |
| `/capture_image` | `std_msgs/Int32` | Trigger image capture (any value) |

### Published Topics
| Topic | Type | Description |
|-------|------|-------------|
| `/crack_detection/result` | `std_msgs/String` | Detection result with metadata |
| `/crack_detection/detected` | `std_msgs/Bool` | Boolean detection flag |
| `/crack_detection/center` | `geometry_msgs/Point` | Crack center pixel coordinates |
| `/crack_detection/robot_pose` | `geometry_msgs/Point` | Robot pose when crack detected |
| `/crack_detection/visualization` | `sensor_msgs/Image` | Annotated visualization image |
| `/crack_detection/marker` | `visualization_msgs/Marker` | RViz marker at crack location |

### Detection Result Message Format
The `/crack_detection/result` topic publishes a comma-separated string:
```
detected=true,crack_percent=5.23,center_x=320,center_y=240,frame=1234,
zoom_enabled=true,zoom_factor=2.0,temporal_status=4/5 frames,
wall_distance=1.45,hysteresis_active=true,timestamp_sec=1234567890,
timestamp_nanosec=123456789
```

## Operation Modes

### 1. Fast Mode (Default)
- **Use Case**: Real-time inspection on moving platforms
- **Performance**: ~20-30 FPS on NVIDIA RTX series
- **Configuration**: Default settings
```bash
ros2 launch crack_detection crack_detection_launch.py
```

### 2. High-Quality Mode
- **Use Case**: Static detailed inspection
- **Performance**: ~5-10 FPS
- **Configuration**: Enable Pix2Pix refinement
```bash
ros2 launch crack_detection crack_detection_launch.py \
    use_pix2pix:=true \
    skip_frames:=3
```

### 3. High-Resolution Mode
- **Use Case**: Large images or high detail requirements
- **Performance**: ~2-5 FPS
- **Configuration**: Enable tiling
```bash
ros2 launch crack_detection crack_detection_launch.py \
    use_tiling:=true \
    window_size:=512 \
    subdivisions:=4
```

### 4. Moving Platform Mode
- **Use Case**: Robotic inspection with motion
- **Performance**: ~15-25 FPS
- **Configuration**: Enhanced temporal filtering + depth filtering
```bash
ros2 launch crack_detection crack_detection_launch.py \
    temporal_window_size:=7 \
    temporal_threshold:=5 \
    depth_filtering_enabled:=true \
    use_adaptive_depth:=true
```

## Image Capture

### Manual Capture
Publish to the capture topic to save the current frame:
```bash
ros2 topic pub --once /capture_image std_msgs/Int32 "{data: 1}"
```

### Saved Files
When a capture is triggered, three images are saved:
- `crack_YYYYMMDD_HHMMSS_original.jpg` - Original camera image
- `crack_YYYYMMDD_HHMMSS_prediction.jpg` - Model prediction heatmap
- `crack_YYYYMMDD_HHMMSS_mask.jpg` - Binary segmentation mask

Files are saved to the directory specified by `save_directory` parameter.

## Visualization

### RViz Setup
1. **Add Image Display**
   - Topic: `/crack_detection/visualization`
   - Shows: Original image + prediction overlay + detection info

2. **Add Marker Display**
   - Topic: `/crack_detection/marker`
   - Shows: Red spheres at crack locations on the map

3. **Add Camera Display**
   - Topics: Camera RGB and depth streams

### Visualization Features
The visualization image shows:
- Original camera feed
- Crack prediction heatmap (blue-red colormap)
- Binary detection mask
- Detection status (DETECTED / CLEAR)
- Crack coverage percentage
- Temporal consistency status
- Wall distance (if depth filtering enabled)
- Robot pose (if AMCL active)
- Zoom information (if enabled)

## Tuning Guide

### Reducing False Positives

1. **Increase detection threshold**
```yaml
min_crack_percent: 5.0  # Require more coverage
threshold: 0.6          # Higher confidence needed
```

2. **Enable temporal consistency**
```yaml
temporal_window_size: 7
temporal_threshold: 6   # 6 out of 7 frames must detect
```

3. **Enable depth filtering**
```yaml
depth_filtering_enabled: true
use_adaptive_depth: true
depth_tolerance: 0.1
```

### Reducing False Negatives

1. **Lower detection threshold**
```yaml
min_crack_percent: 1.0
threshold: 0.4
```

2. **Relax temporal requirements**
```yaml
temporal_window_size: 5
temporal_threshold: 3  # 3 out of 5 frames
```

3. **Enable zoom for small cracks**
```yaml
zoom_enabled: true
zoom_factor: 2.5
```

### Improving Performance

1. **Skip frames**
```yaml
skip_frames: 2  # Process every other frame
```

2. **Reduce input size**
```yaml
input_size: 256  # Lower resolution but faster
```

3. **Disable expensive features**
```yaml
use_pix2pix: false
use_tiling: false
```

## Troubleshooting

### Node doesn't start
- **Check model files exist**
  ```bash
  ls ~/ros2_ws/src/crack_detection/models/
  ```
- **Verify CUDA is available**
  ```bash
  python3 -c "import torch; print(torch.cuda.is_available())"
  ```

### No detection messages
- **Verify camera topics**
  ```bash
  ros2 topic list | grep camera
  ros2 topic echo <camera_topic> --once
  ```
- **Check parameter configuration**
  ```bash
  ros2 param list /crack_detection_node
  ```

### Poor detection accuracy
- **Check model compatibility**: Ensure model was trained with same input size
- **Verify camera calibration**: Poor images lead to poor detection
- **Adjust lighting**: Models perform best with consistent lighting

### High CPU/GPU usage
- **Enable frame skipping**: `skip_frames: 3`
- **Disable visualization**: `publish_visualization: false`
- **Use smaller input size**: `input_size: 256`

### Detection flickering
- **Enable hysteresis**:
  ```yaml
  hysteresis_enabled: true
  hysteresis_low_threshold: 1.0
  hysteresis_min_duration: 0.5
  ```
- **Increase temporal window**:
  ```yaml
  temporal_window_size: 7
  temporal_threshold: 5
  ```

## Performance Benchmarks

### NVIDIA RTX 3060
| Mode | Resolution | FPS | Latency |
|------|-----------|-----|---------|
| Fast (UNet only) | 384×384 | ~25 | 40ms |
| With Pix2Pix | 384×384 | ~8 | 125ms |
| Tiled | 768×768 | ~5 | 200ms |

### NVIDIA Jetson Xavier NX
| Mode | Resolution | FPS | Latency |
|------|-----------|-----|---------|
| Fast (UNet only) | 384×384 | ~12 | 85ms |
| With Pix2Pix | 384×384 | ~4 | 250ms |

*Note: Benchmarks with skip_frames=1, all features enabled*

## Example Launch Commands

### Standard Building Inspection
```bash
ros2 launch crack_detection crack_detection_launch.py \
    camera_topic:=/camera/color/image_raw \
    depth_topic:=/camera/depth/image_raw \
    depth_filtering_enabled:=true \
    use_adaptive_depth:=true \
    temporal_window_size:=5 \
    temporal_threshold:=4 \
    min_crack_percent:=2.5
```

### High-Speed Mobile Inspection
```bash
ros2 launch crack_detection crack_detection_launch.py \
    skip_frames:=2 \
    input_size:=256 \
    temporal_window_size:=7 \
    temporal_threshold:=5 \
    hysteresis_enabled:=true
```

### Detailed Static Inspection
```bash
ros2 launch crack_detection crack_detection_launch.py \
    use_pix2pix:=true \
    use_tiling:=true \
    window_size:=512 \
    subdivisions:=4 \
    skip_frames:=3 \
    zoom_enabled:=true \
    zoom_factor:=3.0
```

## Citation

If you use this package in your research, please cite:
```bibtex
@software{crack_detection_ros2,
  title = {Crack Detection ROS2 Node},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-repo}
}
```

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please submit issues and pull requests on GitHub.

## Contact

For questions and support:
- GitHub Issues: [your-repo-url/issues]
- Email: [your-email]

## Acknowledgments

- UNet architecture based on [Ronneberger et al., 2015]
- Pix2Pix implementation based on [Isola et al., 2017]
- Built with ROS2 Humble and PyTorch

---

**Last Updated**: December 2025
