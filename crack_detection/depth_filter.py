#!/usr/bin/env python3

import cv2
import numpy as np


class DepthFilter:
    """Handles depth-based filtering of crack detections."""
    
    def __init__(self, config, logger):
        """
        Initialize depth filter.
        
        Args:
            config: Configuration object
            logger: ROS2 logger
        """
        self.config = config
        self.logger = logger
    
    def get_dominant_depth(self, depth_image):
        """
        Find the most common depth in the image (likely the wall surface).
        
        Args:
            depth_image: Depth values in meters
            
        Returns:
            float: Dominant depth value (most common distance)
        """
        # Flatten and filter out invalid readings (zeros and very far values)
        valid_depths = depth_image[(depth_image > 0.3) & (depth_image < 5.0)].flatten()
        
        if len(valid_depths) == 0:
            return None
        
        # Use histogram to find most common depth
        hist, bin_edges = np.histogram(valid_depths, bins=50, range=(0.3, 5.0))
        dominant_bin = np.argmax(hist)
        dominant_depth = (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2.0
        
        return dominant_depth
    
    def apply_depth_filter(self, binary_mask, depth_image):
        """
        Filter detections based on depth - only keep pixels at wall surface depth.
        
        Args:
            binary_mask: Binary crack detection mask
            depth_image: Depth values in meters
            
        Returns:
            tuple: (filtered_mask, depth_mask_visualization, wall_distance)
        """
        if depth_image is None or not self.config.depth_filtering_enabled:
            return binary_mask, None, None
        
        # Determine target depth
        if self.config.use_adaptive_depth:
            # Automatically find the wall (most common depth)
            wall_depth = self.get_dominant_depth(depth_image)
            if wall_depth is None:
                self.logger.warn('Could not determine wall depth, skipping depth filter')
                return binary_mask, None, None
        else:
            # Use fixed target distance
            wall_depth = self.config.target_inspection_distance
        
        # Define valid depth range
        min_depth = wall_depth - self.config.depth_tolerance
        max_depth = wall_depth + self.config.depth_tolerance
        
        # Create depth mask (True where depth is in valid range)
        valid_depth_mask = (depth_image >= min_depth) & (depth_image <= max_depth) & (depth_image > 0)
        
        # Convert to uint8 for bitwise operations
        depth_mask_uint8 = valid_depth_mask.astype(np.uint8) * 255
        
        # Apply depth mask to crack detection mask
        filtered_mask = cv2.bitwise_and(binary_mask, binary_mask, mask=depth_mask_uint8)
        
        return filtered_mask, depth_mask_uint8, wall_depth
