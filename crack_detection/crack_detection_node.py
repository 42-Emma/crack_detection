#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch

# Import all modules
from crack_detection.config import Config
from crack_detection.model_manager import ModelManager
from crack_detection.inference_engine import InferenceEngine
from crack_detection.image_processor import ImageProcessor
from crack_detection.depth_filter import DepthFilter
from crack_detection.detection_filters import DetectionFilters
from crack_detection.scan_point_extractor import ScanPointExtractor
from crack_detection.visualization import Visualization
from crack_detection.file_manager import FileManager
from crack_detection.publishers import Publishers
from crack_detection.subscribers import Subscribers


class CrackDetectionNode(Node):
    """ROS2 node for crack detection with UNet and Pix2Pix support."""
    
    def __init__(self):
        super().__init__('crack_detection_node')
        
        # Initialize configuration
        self.config = Config(self)
        self.config.log_configuration()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Device: {self.device}')
        
        # Initialize all components
        self.model_manager = ModelManager(self.config, self.device, self.get_logger())
        self.inference_engine = InferenceEngine(self.model_manager, self.config, self.device)
        self.image_processor = ImageProcessor(self.config)
        self.depth_filter = DepthFilter(self.config, self.get_logger())
        self.detection_filters = DetectionFilters(self.config, self.get_logger())
        self.scan_point_extractor = ScanPointExtractor()
        self.visualization = Visualization(self.config)
        self.file_manager = FileManager(self.config, self.get_logger())
        self.pub_manager = Publishers(self, self.config)
        self.subscribers = Subscribers(self, self.config)
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Frame counting and caching
        self.frame_count = 0
        self.last_pred_prob = None
        self.last_binary_mask = None
        self.last_crack_percentage = 0.0
        self.last_original_frame = None
        
        # Scan points tracking
        self.current_scan_points = None
        
        # Wall distance tracking
        self.current_wall_distance = None
        
        # Set callbacks for subscribers
        self.subscribers.set_image_callback(self.image_callback)
        self.subscribers.set_capture_callback(self.capture_callback)
        
        self.get_logger().info('✓ Initialization complete!')
        self.get_logger().info('⏳ Temporal filter: Collecting initial frames...')
        self.get_logger().info('Waiting for camera images...')
        self.get_logger().info('='*60)
    
    def capture_callback(self, msg):
        """Callback for image capture trigger."""
        if msg.data and self.last_binary_mask is not None and self.last_original_frame is not None:
            self.file_manager.save_capture(
                self.last_original_frame,
                self.last_binary_mask,
                self.last_pred_prob,
                self.subscribers.latest_depth_image,
                self.frame_count,
                self.last_crack_percentage,
                self.current_wall_distance,
                self.model_manager.mode,
                self.current_scan_points  # Add this line
            )
    
    def image_callback(self, msg):
        """Callback function for incoming camera images."""
        self.frame_count += 1
        
        # Convert ROS Image to OpenCV
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
        
        # Store original frame for saving and visualization
        self.last_original_frame = frame_bgr.copy()
        
        # Apply zoom if enabled
        zoomed_frame_bgr, zoom_roi = self.image_processor.apply_zoom(frame_bgr)
        
        # Convert BGR to RGB for model (use zoomed frame)
        frame_rgb = cv2.cvtColor(zoomed_frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Check if we should process this frame
        if self.frame_count % self.config.skip_frames == 1 or self.config.skip_frames == 1:
            # Run inference on zoomed frame
            pred_prob, binary_mask, crack_percentage_raw = self.inference_engine.predict_frame(frame_rgb)
            
            # Apply depth filtering BEFORE calculating final crack percentage
            if self.config.depth_filtering_enabled and self.subscribers.latest_depth_image is not None:
                # Resize depth image to match zoomed frame if needed
                depth_resized = cv2.resize(self.subscribers.latest_depth_image, 
                                          (binary_mask.shape[1], binary_mask.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                
                binary_mask, depth_viz, wall_distance = self.depth_filter.apply_depth_filter(binary_mask, depth_resized)
                self.current_wall_distance = wall_distance
            else:
                depth_viz = None
                self.current_wall_distance = None
            
            # Recalculate crack percentage after depth filtering
            crack_pixels = np.sum(binary_mask > 127)
            total_pixels = binary_mask.size
            crack_percentage = (crack_pixels / total_pixels) * 100
            
            # Cache results
            self.last_pred_prob = pred_prob
            self.last_binary_mask = binary_mask
            self.last_crack_percentage = crack_percentage
        else:
            # Use cached results
            pred_prob = self.last_pred_prob
            binary_mask = self.last_binary_mask
            crack_percentage = self.last_crack_percentage
            depth_viz = None
        
        # Determine current frame detection (single frame decision)
        current_frame_detection = crack_percentage >= self.config.min_crack_percent
        
        # Apply temporal consistency filter
        temporal_detection = self.detection_filters.apply_temporal_filter(current_frame_detection)
        
        # Apply hysteresis to final detection
        is_crack_detected = self.detection_filters.apply_hysteresis(crack_percentage, temporal_detection)
        
        # Create temporal status string for visualization
        detection_count = sum(self.detection_filters.detection_history)
        temporal_status = f"{detection_count}/{len(self.detection_filters.detection_history)} frames"
        
        # Calculate crack center (in zoomed frame coordinates)
        center_x_zoomed, center_y_zoomed = self.scan_point_extractor.calculate_crack_center(binary_mask)
        
        # Map coordinates back to original frame if zoom is enabled
        if self.config.zoom_enabled and zoom_roi is not None:
            center_x, center_y = self.image_processor.map_coordinates_to_original(
                center_x_zoomed, center_y_zoomed, zoom_roi, self.last_original_frame.shape
            )
        else:
            center_x, center_y = center_x_zoomed, center_y_zoomed
        
        # Extract scan points (center, left/start, right/end) if crack detected
        if is_crack_detected:
            scan_points = self.scan_point_extractor.extract_crack_scan_points(binary_mask)
            
            if scan_points is not None:
                # Map scan points back to original frame if zoom is enabled
                if self.config.zoom_enabled and zoom_roi is not None:
                    scan_points_original = {}
                    for point_name, (px, py) in scan_points.items():
                        orig_x, orig_y = self.image_processor.map_coordinates_to_original(
                            px, py, zoom_roi, self.last_original_frame.shape
                        )
                        scan_points_original[point_name] = (orig_x, orig_y)
                else:
                    scan_points_original = scan_points
                    
                # Store for saving (in original frame coordinates)
                self.current_scan_points = scan_points_original
                
                # Log scan points
                self.get_logger().info(
                    f'📍 Scan Points - Center: {scan_points_original["center"]}, '
                    f'Start: {scan_points_original["left_point"]}, '
                    f'End: {scan_points_original["right_point"]}'
                )
                
            else:
                self.current_scan_points = None
        
        # Print detection result
        if is_crack_detected:
            zoom_info = f" [Zoom: {self.config.zoom_factor}x]" if self.config.zoom_enabled else ""
            wall_info = f" [Wall: {self.current_wall_distance:.2f}m]" if self.current_wall_distance else ""
            self.get_logger().info(
                f'🔴 CRACK DETECTED at pixel ({center_x}, {center_y}){zoom_info}{wall_info} | '
                f'Coverage: {crack_percentage:.2f}% | Temporal: {temporal_status} | Frame: {self.frame_count}'
            )
            # Log AMCL pose on separate line
            if self.subscribers.current_pose is not None:
                self.get_logger().info(
                    f'📍 Robot Pose: x={self.subscribers.current_pose.x:.3f}m, '
                    f'y={self.subscribers.current_pose.y:.3f}m, z={self.subscribers.current_pose.z:.3f}m'
                )
        
        # Get timestamp from message header
        timestamp_sec = msg.header.stamp.sec
        timestamp_nanosec = msg.header.stamp.nanosec
        
        # Publish all results
        self.pub_manager.publish_detection_result(
            is_crack_detected, crack_percentage, center_x, center_y,
            self.frame_count, temporal_status, self.current_wall_distance,
            self.detection_filters.currently_in_detection,
            timestamp_sec, timestamp_nanosec
        )
        
        self.pub_manager.publish_boolean_detection(is_crack_detected)
        
        if is_crack_detected:
            self.pub_manager.publish_crack_center(center_x, center_y)
            self.pub_manager.publish_robot_pose(self.subscribers.current_pose)
            self.pub_manager.publish_marker(self.subscribers.current_pose, self.frame_count)
            self.pub_manager.publish_scan_points_2d(self.current_scan_points)
        
        # Publish visualization
        if self.config.publish_visualization:
            viz_image = self.visualization.create_visualization(
                zoomed_frame_bgr, pred_prob, binary_mask, crack_percentage,
                is_crack_detected, zoom_roi, temporal_status, depth_viz,
                self.current_wall_distance, self.last_original_frame,
                self.model_manager.mode, self.frame_count,
                self.detection_filters.currently_in_detection
            )
            
            self.pub_manager.publish_visualization(viz_image, msg.header, self.bridge)
            
            # Publish scan points visualization (only when crack detected)
            if is_crack_detected and self.current_scan_points is not None:
                scan_viz_image = self.visualization.create_scan_points_visualization(
                        self.last_original_frame, binary_mask, self.current_scan_points
                    )
                self.pub_manager.publish_scan_visualization(scan_viz_image, msg.header, self.bridge)


def main(args=None):
    rclpy.init(args=args)
    
    node = CrackDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down crack detection node...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
