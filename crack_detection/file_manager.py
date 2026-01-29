#!/usr/bin/env python3

import os
import cv2
from datetime import datetime


class FileManager:
    """Handles file I/O operations for saving captures."""
    
    def __init__(self, config, logger):
        """
        Initialize file manager.
        
        Args:
            config: Configuration object
            logger: ROS2 logger
        """
        self.config = config
        self.logger = logger
    
    def save_capture(self, original_frame, binary_mask, pred_prob, depth_image, 
                frame_count, crack_percentage, wall_distance, mode, scan_points=None):
        """
        Save captured images and data.
        
        Args:
            original_frame: Original BGR frame
            binary_mask: Binary crack mask
            pred_prob: Probability map
            depth_image: Depth image (can be None)
            frame_count: Current frame count
            crack_percentage: Crack coverage percentage
            wall_distance: Wall distance (can be None)
            mode: Current detection mode
        """
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Save original frame
        original_path = os.path.join(self.config.save_directory, f"frame_{timestamp}.png")
        cv2.imwrite(original_path, original_frame)
        
        # Save binary mask
        mask_path = os.path.join(self.config.save_directory, f"mask_{timestamp}.png")
        cv2.imwrite(mask_path, binary_mask)
        
        # Save probability map as heatmap
        if pred_prob is not None:
            heatmap = (pred_prob * 255).astype('uint8')
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap_path = os.path.join(self.config.save_directory, f"heatmap_{timestamp}.png")
            cv2.imwrite(heatmap_path, heatmap_colored)
        
        # Save depth image if available
        if depth_image is not None:
            import numpy as np
            depth_normalized = np.clip(depth_image / 3.0, 0, 1) * 255
            depth_path = os.path.join(self.config.save_directory, f"depth_{timestamp}.png")
            cv2.imwrite(depth_path, depth_normalized.astype('uint8'))
        
        # Save visualization with scan points if available
        if scan_points is not None:
            viz_frame = original_frame.copy()
            
            # Draw crack overlay
            crack_pixels = binary_mask > 127
            overlay = viz_frame.copy()
            overlay[crack_pixels] = [0, 0, 255]
            viz_frame = cv2.addWeighted(viz_frame, 0.7, overlay, 0.3, 0)
            
            # Draw scan points
            center_x, center_y = scan_points['center']
            left_x, left_y = scan_points['left_point']
            right_x, right_y = scan_points['right_point']
            
            cv2.circle(viz_frame, (center_x, center_y), 10, (255, 0, 255), -1)
            cv2.circle(viz_frame, (left_x, left_y), 10, (0, 255, 255), -1)
            cv2.circle(viz_frame, (right_x, right_y), 10, (0, 255, 255), -1)
            
            cv2.putText(viz_frame, "CENTER", (center_x + 15, center_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(viz_frame, "START", (left_x + 15, left_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(viz_frame, "END", (right_x + 15, right_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            viz_path = os.path.join(self.config.save_directory, f"visualization_{timestamp}.png")
            cv2.imwrite(viz_path, viz_frame)
        
        # Save info text file
        info_path = os.path.join(self.config.save_directory, f"info_{timestamp}.txt")
        with open(info_path, 'w') as f:
            f.write(f"Frame: {frame_count}\n")
            f.write(f"Crack Percentage: {crack_percentage:.2f}%\n")
            f.write(f"Wall Distance: {wall_distance:.2f}m\n" if wall_distance else "Wall Distance: N/A\n")
            f.write(f"Zoom Enabled: {self.config.zoom_enabled}\n")
            f.write(f"Zoom Factor: {self.config.zoom_factor}\n")
            f.write(f"Depth Filtering: {self.config.depth_filtering_enabled}\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Threshold: {self.config.threshold}\n")
        
        self.logger.info(f'📸 Captured images saved to {self.config.save_directory} with timestamp {timestamp}')
