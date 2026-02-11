#!/usr/bin/env python3

import numpy as np
import cv2


class ScanPointExtractor:
    """Extracts 2D pixel scan points from crack detections."""
    
    def __init__(self):
        """Initialize scan point extractor."""
        pass
    
    def extract_crack_scan_points(self, binary_mask):
        """
        Extract 3 key points from the crack mask for scanning.
        LEGACY METHOD - kept for backward compatibility.
        
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
        LEGACY METHOD - kept for backward compatibility.
        
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
    
    def extract_all_cracks(self, binary_mask):
        """
        Extract all individual crack blobs from the binary mask using connected components.
        
        Args:
            binary_mask: Binary crack detection mask
            
        Returns:
            list: List of dictionaries, each containing:
                - 'id': Unique blob ID
                - 'center': (x, y) tuple
                - 'left_point': (x, y) tuple (leftmost point)
                - 'right_point': (x, y) tuple (rightmost point)
                - 'area': Number of pixels
                - 'bbox': (x, y, width, height) bounding box
                - 'aspect_ratio': width/height ratio
            Returns empty list if no cracks detected
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        crack_list = []
        
        # Skip label 0 (background)
        for label_id in range(1, num_labels):
            # Extract stats for this component
            x, y, width, height, area = stats[label_id]
            
            # Create mask for just this component
            component_mask = (labels == label_id).astype(np.uint8) * 255
            
            # Get all pixels for this component
            component_pixels = np.where(component_mask > 0)
            y_coords = component_pixels[0]
            x_coords = component_pixels[1]
            
            # Calculate center (centroid from OpenCV is already good)
            center_x = int(centroids[label_id][0])
            center_y = int(centroids[label_id][1])
            
            # Find leftmost and rightmost points
            leftmost_idx = np.argmin(x_coords)
            left_x = int(x_coords[leftmost_idx])
            left_y = int(y_coords[leftmost_idx])
            
            rightmost_idx = np.argmax(x_coords)
            right_x = int(x_coords[rightmost_idx])
            right_y = int(y_coords[rightmost_idx])
            
            # Calculate aspect ratio (avoid division by zero)
            if height > 0:
                aspect_ratio = width / height
            else:
                aspect_ratio = 0.0
            
            # Extract crack angle using minAreaRect
            rect = cv2.minAreaRect(np.column_stack((x_coords, y_coords)))
            angle = rect[2]  # Angle is the third element of the tuple
            
            # Store crack info
            crack_info = {
                'id': label_id,
                'center': (center_x, center_y),
                'left_point': (left_x, left_y),
                'right_point': (right_x, right_y),
                'area': area,
                'bbox': (x, y, width, height),
                'aspect_ratio': aspect_ratio,
                'angle': angle  # NEW LINE - add the angle
            }
            
            crack_list.append(crack_info)
        
        return crack_list
    
    def get_crack_endpoints(self, binary_mask):
        """
        Find the two endpoints of a crack blob (topmost and bottommost points).
        Useful for vertical or diagonal cracks.
        
        Args:
            binary_mask: Binary mask of a single crack
            
        Returns:
            tuple: ((top_x, top_y), (bottom_x, bottom_y)) or None
        """
        crack_pixels = np.where(binary_mask > 127)
        
        if len(crack_pixels[0]) == 0:
            return None
        
        y_coords = crack_pixels[0]
        x_coords = crack_pixels[1]
        
        # Find topmost point (minimum Y)
        topmost_idx = np.argmin(y_coords)
        top_x = int(x_coords[topmost_idx])
        top_y = int(y_coords[topmost_idx])
        
        # Find bottommost point (maximum Y)
        bottommost_idx = np.argmax(y_coords)
        bottom_x = int(x_coords[bottommost_idx])
        bottom_y = int(y_coords[bottommost_idx])
        
        return (top_x, top_y), (bottom_x, bottom_y)
