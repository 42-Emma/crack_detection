#!/usr/bin/env python3

from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker


class Publishers:
    """Manages all ROS2 publishers."""
    
    def __init__(self, node, config):
        """
        Initialize publishers.
        
        Args:
            node: ROS2 node instance
            config: Configuration object
        """
        self.node = node
        self.config = config
        
        # Create all publishers
        self._create_publishers()
    
    def _create_publishers(self):
        """Create all ROS2 publishers."""
        self.detection_pub = self.node.create_publisher(
            String,
            '/crack_detection/result',
            10
        )
        
        self.crack_detected_pub = self.node.create_publisher(
            Bool,
            '/crack_detection/detected',
            10
        )
        
        self.crack_center_pub = self.node.create_publisher(
            Point,
            '/crack_detection/center_pixel',
            10
        )
        
        # Publisher for crack location markers in RViz
        self.marker_pub = self.node.create_publisher(
            Marker,
            '/visualization_marker',
            10
        )
        
        # Publisher for robot pose when crack is detected
        self.robot_pose_pub = self.node.create_publisher(
            Point,
            '/crack_detection/robot_pose',
            10
        )
        
        if self.config.publish_visualization:
            self.viz_pub = self.node.create_publisher(
                Image,
                '/crack_detection/visualization',
                10
            )
            
            # Publisher for scan points visualization
            self.scan_viz_pub = self.node.create_publisher(
                Image,
                '/crack_detection/scan_visualization',
                10
            )
    
    def publish_detection_result(self, is_crack_detected, crack_percentage, center_x, center_y, 
                                frame_count, temporal_status, wall_distance, currently_in_detection,
                                timestamp_sec, timestamp_nanosec):
        """Publish detection result with all metadata."""
        result_msg = String()
        result_msg.data = (f'detected={is_crack_detected},crack_percent={crack_percentage:.2f},'
                          f'center_x={center_x},center_y={center_y},frame={frame_count},'
                          f'zoom_enabled={self.config.zoom_enabled},zoom_factor={self.config.zoom_factor},'
                          f'temporal_status={temporal_status},'
                          f'wall_distance={wall_distance if wall_distance else 0.0},'
                          f'hysteresis_active={currently_in_detection},'
                          f'timestamp_sec={timestamp_sec},timestamp_nanosec={timestamp_nanosec}')
        self.detection_pub.publish(result_msg)
    
    def publish_boolean_detection(self, is_crack_detected):
        """Publish boolean detection."""
        detected_msg = Bool()
        detected_msg.data = bool(is_crack_detected)
        self.crack_detected_pub.publish(detected_msg)
    
    def publish_crack_center(self, center_x, center_y):
        """Publish crack center point."""
        center_msg = Point()
        center_msg.x = float(center_x)
        center_msg.y = float(center_y)
        center_msg.z = 0.0
        self.crack_center_pub.publish(center_msg)
    
    def publish_robot_pose(self, pose):
        """Publish robot pose when crack is detected."""
        if pose is not None:
            pose_msg = Point()
            pose_msg.x = pose.x
            pose_msg.y = pose.y
            pose_msg.z = pose.z
            self.robot_pose_pub.publish(pose_msg)
    
    def publish_marker(self, pose, frame_count):
        """Publish red marker at robot's current pose when crack is detected."""
        if pose is not None:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.node.get_clock().now().to_msg()
            marker.ns = 'crack_locations'
            marker.id = frame_count  # Use frame count for unique IDs
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = pose.x
            marker.pose.position.y = pose.y
            marker.pose.position.z = 0.1  # Slightly above ground
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime.sec = 0  # Marker persists forever
            self.marker_pub.publish(marker)
    
    def publish_visualization(self, viz_image, header, bridge):
        """Publish visualization image."""
        if self.config.publish_visualization:
            try:
                viz_msg = bridge.cv2_to_imgmsg(viz_image, encoding='bgr8')
                viz_msg.header = header
                self.viz_pub.publish(viz_msg)
            except Exception as e:
                self.node.get_logger().error(f'Failed to publish visualization: {e}')
                
    def publish_scan_visualization(self, scan_viz_image, header, bridge):
        """Publish scan points visualization image."""
        if self.config.publish_visualization:
            try:
                scan_viz_msg = bridge.cv2_to_imgmsg(scan_viz_image, encoding='bgr8')
                scan_viz_msg.header = header
                self.scan_viz_pub.publish(scan_viz_msg)
            except Exception as e:
                self.node.get_logger().error(f'Failed to publish scan visualization: {e}')
