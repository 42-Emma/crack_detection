#!/usr/bin/env python3

import numpy as np


class ScanPointExtractor:
    """Extracts 2D pixel scan points from crack detections."""
    
    def __init__(self):
        """Initialize scan point extractor."""
        pass
    
    def extract_crack_scan_points(self, binary_mask):
        """
        Extract 3 key points from the crack mask for scanning.
        
        Args:
            binary_mask: Binary crack detection mask
        
        Returns:
            dict: Contains 'center', 'left_point', 'right_point' as (x, y) tuples
                  Returns None if no crack
        """
        crack_pixels = np.where(binary_mask > 127)
        
        if len(crack_pixels[0]) == 0:
            return None
        
        # Get all crack pixel coordinates
        y_coords = crack_pixels[0]  # Row indices
        x_coords = crack_pixels[1]  # Column indices
        
        # Calculate center point (average of all crack pixels)
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        
        # Find leftmost point (minimum X coordinate)
        leftmost_idx = np.argmin(x_coords)
        left_x = int(x_coords[leftmost_idx])
        left_y = int(y_coords[leftmost_idx])
        
        # Find rightmost point (maximum X coordinate)
        rightmost_idx = np.argmax(x_coords)
        right_x = int(x_coords[rightmost_idx])
        right_y = int(y_coords[rightmost_idx])
        
        return {
            'center': (center_x, center_y),
            'left_point': (left_x, left_y),
            'right_point': (right_x, right_y)
        }
    
    def calculate_crack_center(self, binary_mask):
        """
        Calculate the center point of detected cracks.
        
        Args:
            binary_mask: Binary crack detection mask
            
        Returns:
            tuple: (center_x, center_y)
        """
        crack_pixels = np.where(binary_mask > 127)
        
        if len(crack_pixels[0]) > 0:
            center_y = int(np.mean(crack_pixels[0]))
            center_x = int(np.mean(crack_pixels[1]))
            return center_x, center_y
        else:
            return 0, 0
