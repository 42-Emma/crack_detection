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
        
        # Multi-crack tracking
        self.current_crack_list = []  # List of all valid cracks in current frame
        self.primary_crack = None     # Largest crack (for legacy compatibility)
        
        # Wall distance tracking
        self.current_wall_distance = None
        
        # Set callbacks for subscribers
        self.subscribers.set_image_callback(self.image_callback)
        self.subscribers.set_capture_callback(self.capture_callback)
        
        self.get_logger().info('✔ Initialization complete!')
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
                self.primary_crack  # Save primary crack scan points
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
            
            # Apply depth filtering BEFORE extracting individual cracks
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
            
            # NEW: Extract ALL individual crack blobs
            all_cracks = self.scan_point_extractor.extract_all_cracks(binary_mask)
            
            # NEW: Apply geometric validation to filter out shadows/wires
            valid_cracks = self.detection_filters.validate_crack_list(all_cracks)
            
            # NEW: Sort by area (largest first)
            valid_cracks.sort(key=lambda c: c['area'], reverse=True)
            
            # NEW: Limit to max cracks per frame
            if len(valid_cracks) > self.config.max_cracks_per_frame:
                valid_cracks = valid_cracks[:self.config.max_cracks_per_frame]
            
            # NEW: Map coordinates back to original frame if zoom is enabled
            if self.config.zoom_enabled and zoom_roi is not None:
                for crack in valid_cracks:
                    # Map center
                    center_x, center_y = self.image_processor.map_coordinates_to_original(
                        crack['center'][0], crack['center'][1], zoom_roi, self.last_original_frame.shape
                    )
                    crack['center'] = (center_x, center_y)
                    
                    # Map left point
                    left_x, left_y = self.image_processor.map_coordinates_to_original(
                        crack['left_point'][0], crack['left_point'][1], zoom_roi, self.last_original_frame.shape
                    )
                    crack['left_point'] = (left_x, left_y)
                    
                    # Map right point
                    right_x, right_y = self.image_processor.map_coordinates_to_original(
                        crack['right_point'][0], crack['right_point'][1], zoom_roi, self.last_original_frame.shape
                    )
                    crack['right_point'] = (right_x, right_y)
            
            # Store results
            self.current_crack_list = valid_cracks
            self.primary_crack = valid_cracks[0] if len(valid_cracks) > 0 else None
            
            # Cache inference results
            self.last_pred_prob = pred_prob
            self.last_binary_mask = binary_mask
            self.last_crack_percentage = crack_percentage
        else:
            # Use cached results
            pred_prob = self.last_pred_prob
            binary_mask = self.last_binary_mask
            crack_percentage = self.last_crack_percentage
            depth_viz = None
            # Keep previously detected cracks
        
        # Determine current frame detection (any valid crack found)
        current_frame_detection = len(self.current_crack_list) > 0
        
        # Apply temporal consistency filter
        temporal_detection = self.detection_filters.apply_temporal_filter(current_frame_detection)
        
        # Apply hysteresis to final detection
        is_crack_detected = self.detection_filters.apply_hysteresis(crack_percentage, temporal_detection)
        
        # Create temporal status string for visualization
        detection_count = sum(self.detection_filters.detection_history)
        temporal_status = f"{detection_count}/{len(self.detection_filters.detection_history)} frames"
        
        # Get primary crack info for legacy publishers
        if self.primary_crack is not None:
            center_x, center_y = self.primary_crack['center']
        else:
            center_x, center_y = 0, 0
        
        # Print detection result
        if is_crack_detected and len(self.current_crack_list) > 0:
            zoom_info = f" [Zoom: {self.config.zoom_factor}x]" if self.config.zoom_enabled else ""
            wall_info = f" [Wall: {self.current_wall_distance:.2f}m]" if self.current_wall_distance else ""
            self.get_logger().info(
                f'🔴 {len(self.current_crack_list)} CRACK(S) DETECTED at pixel ({center_x}, {center_y}){zoom_info}{wall_info} | '
                f'Coverage: {crack_percentage:.2f}% | Temporal: {temporal_status} | Frame: {self.frame_count}'
            )
            # Log details of each crack
            for i, crack in enumerate(self.current_crack_list):
                self.get_logger().info(
                    f'   Crack {i+1}: Area={crack["area"]}px, '
                    f'Center={crack["center"]}, '
                    f'Start={crack["left_point"]}, End={crack["right_point"]}'
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
            # Legacy single-crack publishers (publish primary/largest crack)
            self.pub_manager.publish_crack_center(center_x, center_y)
            self.pub_manager.publish_robot_pose(self.subscribers.current_pose)
            self.pub_manager.publish_marker(self.subscribers.current_pose, self.frame_count)
            
            if self.primary_crack is not None:
                self.pub_manager.publish_scan_points_2d(self.primary_crack)
            
            # NEW: Multi-crack publisher (all valid cracks)
            self.pub_manager.publish_crack_list(self.current_crack_list)
            
            # NEW: Publish crack angles for motion control
            self.pub_manager.publish_crack_angles(self.current_crack_list)  # ADD THIS LINE
        
        # Publish visualization (unchanged - same as before)
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
            if is_crack_detected and self.primary_crack is not None:
                scan_viz_image = self.visualization.create_scan_points_visualization(
                    self.last_original_frame, binary_mask, self.primary_crack
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
