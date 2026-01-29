#!/usr/bin/env python3

from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseWithCovarianceStamped


class Subscribers:
    """Manages all ROS2 subscribers."""
    
    def __init__(self, node, config):
        """
        Initialize subscribers.
        
        Args:
            node: ROS2 node instance
            config: Configuration object
        """
        self.node = node
        self.config = config
        
        # Storage for latest data
        self.latest_depth_image = None
        self.current_pose = None
        
        # Create all subscribers
        self._create_subscribers()
    
    def _create_subscribers(self):
        """Create all ROS2 subscribers."""
        # Image subscriber (callback will be set by main node)
        self.image_sub = None
        
        # Depth subscriber
        self.depth_sub = self.node.create_subscription(
            Image,
            self.config.depth_topic,
            self.depth_callback,
            10
        )
        
        # Capture trigger subscriber (callback will be set by main node)
        self.capture_sub = None
        
        # AMCL pose subscriber
        self.pose_sub = self.node.create_subscription(
            PoseWithCovarianceStamped,
            '/a200_1103/amcl_pose',
            self.pose_callback,
            10
        )
    
    def set_image_callback(self, callback):
        """Set the image callback function."""
        self.image_sub = self.node.create_subscription(
            Image,
            self.config.camera_topic,
            callback,
            10
        )
    
    def set_capture_callback(self, callback):
        """Set the capture callback function."""
        self.capture_sub = self.node.create_subscription(
            Bool,
            self.config.capture_key_topic,
            callback,
            10
        )
    
    def depth_callback(self, msg):
        """Callback for depth image messages."""
        import numpy as np
        from cv_bridge import CvBridge
        
        bridge = CvBridge()
        
        try:
            # Convert depth image to numpy array
            depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # Check encoding and convert to meters if needed
            if msg.encoding == '16UC1':  # Depth in millimeters
                self.latest_depth_image = depth_image.astype(np.float32) / 1000.0
            elif msg.encoding == '32FC1':  # Depth already in meters
                self.latest_depth_image = depth_image.astype(np.float32)
            else:
                self.node.get_logger().warn(f'Unknown depth encoding: {msg.encoding}')
                self.latest_depth_image = depth_image.astype(np.float32)
                
        except Exception as e:
            self.node.get_logger().error(f'Failed to convert depth image: {e}')
    
    def pose_callback(self, msg):
        """Callback for AMCL pose messages."""
        # Store the latest pose
        self.current_pose = msg.pose.pose.position
