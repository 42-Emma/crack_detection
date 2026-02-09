#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory


class Config:
    """Configuration manager for crack detection node."""
    
    def __init__(self, node):
        """
        Initialize configuration from ROS2 parameters.
        
        Args:
            node: ROS2 node instance
        """
        self.node = node
        
        # Declare all parameters
        self._declare_parameters()
        
        # Get and process parameters
        self._load_parameters()
    
    def _declare_parameters(self):
        """Declare all ROS2 parameters."""
        self.node.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', 'models/best_model.pth'),
                ('pix2pix_model_path', 'models/pix2pix_epoch_98_best.pth'),
                ('input_size', 384),
                ('threshold', 0.5),
                ('min_crack_percent', 5.0),
                ('use_tiling', False),
                ('use_pix2pix', False),
                ('window_size', 384),
                ('subdivisions', 2),
                ('skip_frames', 1),
                ('camera_topic', '/camera/color/image_raw'),
                ('depth_topic', '/camera/depth/image'),
                ('publish_visualization', True),
                # Zoom parameters
                ('zoom_enabled', False),
                ('zoom_factor', 2.0),
                ('zoom_center_x', 0.5),
                ('zoom_center_y', 0.5),
                # Image capture parameters
                ('save_directory', 'crack_captures'),
                ('capture_key_topic', '/capture_image'),
                # Temporal consistency parameters
                ('temporal_window_size', 5),
                ('temporal_threshold', 4),
                # Hysteresis parameters
                ('hysteresis_enabled', True),
                ('hysteresis_low_threshold', 3.0),
                ('hysteresis_min_duration', 0.5),
                # Depth filtering parameters
                ('depth_filtering_enabled', True),
                ('use_adaptive_depth', True),
                ('target_inspection_distance', 1.5),
                ('depth_tolerance', 0.1),
                # Multi-crack detection parameters (NEW)
                ('min_crack_area', 50),
                ('min_aspect_ratio', 2.0),
                ('max_aspect_ratio', 15.0),
                ('max_cracks_per_frame', 10),
            ]
        )
    
    def _load_parameters(self):
        """Load and process all parameters."""
        # Model paths
        model_path_param = self.node.get_parameter('model_path').value
        pix2pix_path_param = self.node.get_parameter('pix2pix_model_path').value
        
        # Convert relative paths to absolute using package share directory
        if not os.path.isabs(model_path_param):
            package_share = get_package_share_directory('crack_detection')
            self.model_path = os.path.join(package_share, model_path_param)
        else:
            self.model_path = model_path_param
            
        if not os.path.isabs(pix2pix_path_param):
            package_share = get_package_share_directory('crack_detection')
            self.pix2pix_model_path = os.path.join(package_share, pix2pix_path_param)
        else:
            self.pix2pix_model_path = pix2pix_path_param
        
        # Model parameters
        self.input_size = self.node.get_parameter('input_size').value
        self.threshold = self.node.get_parameter('threshold').value
        self.min_crack_percent = self.node.get_parameter('min_crack_percent').value
        self.use_tiling = self.node.get_parameter('use_tiling').value
        self.use_pix2pix = self.node.get_parameter('use_pix2pix').value
        self.window_size = self.node.get_parameter('window_size').value
        self.subdivisions = self.node.get_parameter('subdivisions').value
        self.skip_frames = self.node.get_parameter('skip_frames').value
        
        # Topic names
        self.camera_topic = self.node.get_parameter('camera_topic').value
        self.depth_topic = self.node.get_parameter('depth_topic').value
        self.publish_visualization = self.node.get_parameter('publish_visualization').value
        
        # Zoom parameters
        self.zoom_enabled = self.node.get_parameter('zoom_enabled').value
        self.zoom_factor = self.node.get_parameter('zoom_factor').value
        self.zoom_center_x = self.node.get_parameter('zoom_center_x').value
        self.zoom_center_y = self.node.get_parameter('zoom_center_y').value
        
        # Capture parameters
        self.save_directory = self.node.get_parameter('save_directory').value
        self.capture_key_topic = self.node.get_parameter('capture_key_topic').value
        
        # Temporal consistency parameters
        self.temporal_window_size = self.node.get_parameter('temporal_window_size').value
        self.temporal_threshold = self.node.get_parameter('temporal_threshold').value
        
        # Hysteresis parameters
        self.hysteresis_enabled = self.node.get_parameter('hysteresis_enabled').value
        self.hysteresis_low_threshold = self.node.get_parameter('hysteresis_low_threshold').value
        self.hysteresis_min_duration = self.node.get_parameter('hysteresis_min_duration').value
        
        # Depth filtering parameters
        self.depth_filtering_enabled = self.node.get_parameter('depth_filtering_enabled').value
        self.use_adaptive_depth = self.node.get_parameter('use_adaptive_depth').value
        self.target_inspection_distance = self.node.get_parameter('target_inspection_distance').value
        self.depth_tolerance = self.node.get_parameter('depth_tolerance').value
        
        # Multi-crack detection parameters (NEW)
        self.min_crack_area = self.node.get_parameter('min_crack_area').value
        self.min_aspect_ratio = self.node.get_parameter('min_aspect_ratio').value
        self.max_aspect_ratio = self.node.get_parameter('max_aspect_ratio').value
        self.max_cracks_per_frame = self.node.get_parameter('max_cracks_per_frame').value
        
        # Create save directory if it doesn't exist
        if not os.path.isabs(self.save_directory):
            self.save_directory = os.path.join(os.getcwd(), self.save_directory)
        os.makedirs(self.save_directory, exist_ok=True)
    
    def log_configuration(self):
        """Log all configuration settings."""
        self.node.get_logger().info('='*60)
        self.node.get_logger().info('Crack Detection Node Initializing...')
        self.node.get_logger().info('='*60)
        self.node.get_logger().info(f'UNet model path: {self.model_path}')
        if self.use_pix2pix:
            self.node.get_logger().info(f'Pix2Pix model path: {self.pix2pix_model_path}')
        self.node.get_logger().info(f'Detection threshold: {self.threshold}')
        self.node.get_logger().info(f'Minimum crack %: {self.min_crack_percent}%')
        self.node.get_logger().info(f'Frame skipping: Process every {self.skip_frames} frame(s)')
        self.node.get_logger().info(f'Camera topic: {self.camera_topic}')
        self.node.get_logger().info(f'Depth topic: {self.depth_topic}')
        
        # Log temporal consistency settings
        self.node.get_logger().info(f'Temporal window size: {self.temporal_window_size} frames')
        self.node.get_logger().info(f'Temporal threshold: {self.temporal_threshold}/{self.temporal_window_size} frames must detect')
        
        # Log hysteresis settings
        if self.hysteresis_enabled:
            self.node.get_logger().info(f'Hysteresis enabled: Low threshold={self.hysteresis_low_threshold}%, Min duration={self.hysteresis_min_duration}s')
        else:
            self.node.get_logger().info('Hysteresis disabled')
        
        # Log depth filtering settings
        if self.depth_filtering_enabled:
            if self.use_adaptive_depth:
                self.node.get_logger().info(f'Depth filtering: ADAPTIVE (±{self.depth_tolerance}m tolerance)')
            else:
                self.node.get_logger().info(f'Depth filtering: FIXED ({self.target_inspection_distance}m ±{self.depth_tolerance}m)')
        else:
            self.node.get_logger().info('Depth filtering disabled')
        
        # Log zoom settings
        if self.zoom_enabled:
            self.node.get_logger().info(f'Zoom enabled: {self.zoom_factor}x at ({self.zoom_center_x:.2f}, {self.zoom_center_y:.2f})')
        else:
            self.node.get_logger().info('Zoom disabled')
        
        # Log multi-crack detection settings (NEW)
        self.node.get_logger().info(f'Multi-crack detection: Min area={self.min_crack_area}px, '
                                    f'Aspect ratio=[{self.min_aspect_ratio:.1f}, {self.max_aspect_ratio:.1f}], '
                                    f'Max cracks={self.max_cracks_per_frame}')
        
        self.node.get_logger().info(f'Save directory: {self.save_directory}')
