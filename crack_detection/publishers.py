#!/usr/bin/env python3

from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Point, Pose, PoseArray
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
        
        # Legacy single crack publisher (publishes largest crack only)
        self.crack_center_pub = self.node.create_publisher(
            Point,
            '/crack_detection/center_pixel',
            10
        )
        
        # NEW: Multi-crack publisher (all detected cracks)
        self.crack_list_pub = self.node.create_publisher(
            PoseArray,
            '/crack_detection/crack_list',
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
        
        # Publisher for 2D scan points (legacy - single crack)
        self.scan_points_2d_pub = self.node.create_publisher(
            String,
            '/crack_detection/scan_points_2d',
            10
        )
        
        # Publishers for scan points (2D pixel + 3D coordinates)
        self.scan_center_pub = self.node.create_publisher(
            Point,
            '/crack_detection/scan_center',
            10
        )
        
        self.scan_start_pub = self.node.create_publisher(
            Point,
            '/crack_detection/scan_start',
            10
        )
        
        self.scan_end_pub = self.node.create_publisher(
            Point,
            '/crack_detection/scan_end',
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
        """Publish crack center point (legacy - single crack)."""
        center_msg = Point()
        center_msg.x = float(center_x)
        center_msg.y = float(center_y)
        center_msg.z = 0.0
        self.crack_center_pub.publish(center_msg)
    
    def publish_crack_list(self, crack_list):
        """
        Publish array of all detected cracks.
        
        Args:
            crack_list (list): List of crack dictionaries with 'center', 'left_point', 
                              'right_point', 'area', 'id' keys
        """
        pose_array = PoseArray()
        pose_array.header.stamp = self.node.get_clock().now().to_msg()
        pose_array.header.frame_id = 'camera_color_optical_frame'
        
        for crack in crack_list:
            pose = Pose()
            # Position = center point (2D pixel coordinates, z=0)
            pose.position.x = float(crack['center'][0])
            pose.position.y = float(crack['center'][1])
            pose.position.z = float(crack['area'])  # Store area in z for now (metadata)
            
            # Orientation = placeholder (will be used for crack angle in Phase 2)
            pose.orientation.x = float(crack['left_point'][0])  # Store left point in orientation.x
            pose.orientation.y = float(crack['left_point'][1])  # Store left point in orientation.y
            pose.orientation.z = float(crack['right_point'][0])  # Store right point in orientation.z
            pose.orientation.w = float(crack['right_point'][1])  # Store right point in orientation.w
            
            pose_array.poses.append(pose)
        
        self.crack_list_pub.publish(pose_array)
    
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
    
    def publish_scan_points(self, scan_points_2d, scan_points_3d=None):
        """
        Publish scan points (center, start, end) with both 2D pixel and 3D coordinates.
        
        Args:
            scan_points_2d: Dictionary with 'center', 'left_point', 'right_point' as (x, y) tuples
            scan_points_3d: Optional dictionary with 3D coordinates (x, y, z) in meters
        """
        if scan_points_2d is None:
            return
        
        # Publish center point
        center_msg = Point()
        if scan_points_3d is not None and 'center' in scan_points_3d:
            # 3D coordinates available
            center_msg.x, center_msg.y, center_msg.z = scan_points_3d['center']
        else:
            # Only 2D available
            center_msg.x = float(scan_points_2d['center'][0])
            center_msg.y = float(scan_points_2d['center'][1])
            center_msg.z = 0.0
        self.scan_center_pub.publish(center_msg)
        
        # Publish start (left) point
        start_msg = Point()
        if scan_points_3d is not None and 'left_point' in scan_points_3d:
            start_msg.x, start_msg.y, start_msg.z = scan_points_3d['left_point']
        else:
            start_msg.x = float(scan_points_2d['left_point'][0])
            start_msg.y = float(scan_points_2d['left_point'][1])
            start_msg.z = 0.0
        self.scan_start_pub.publish(start_msg)
        
        # Publish end (right) point
        end_msg = Point()
        if scan_points_3d is not None and 'right_point' in scan_points_3d:
            end_msg.x, end_msg.y, end_msg.z = scan_points_3d['right_point']
        else:
            end_msg.x = float(scan_points_2d['right_point'][0])
            end_msg.y = float(scan_points_2d['right_point'][1])
            end_msg.z = 0.0
        self.scan_end_pub.publish(end_msg)
    
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
    
    def publish_scan_points_2d(self, scan_points):
        """
        Publish 2D scan points as a single string message (legacy - single crack).
        Format: "center:x,y;start:x,y;end:x,y"
        
        Args:
            scan_points: Dictionary with 'center', 'left_point', 'right_point' as (x, y) tuples
        """
        if scan_points is not None:
            center_x, center_y = scan_points['center']
            start_x, start_y = scan_points['left_point']
            end_x, end_y = scan_points['right_point']
            
            msg = String()
            msg.data = f"center:{center_x},{center_y};start:{start_x},{start_y};end:{end_x},{end_y}"
            self.scan_points_2d_pub.publish(msg)
