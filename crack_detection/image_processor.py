#!/usr/bin/env python3

import cv2


class ImageProcessor:
    """Handles image preprocessing and zoom operations."""
    
    def __init__(self, config):
        """
        Initialize image processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.zoom_roi = None
    
    def calculate_zoom_roi(self, frame_shape):
        """Calculate the Region of Interest for zooming."""
        height, width = frame_shape[:2]
        
        # Calculate new dimensions after zoom
        new_width = int(width / self.config.zoom_factor)
        new_height = int(height / self.config.zoom_factor)
        
        # Calculate center point in pixels
        center_x_px = int(width * self.config.zoom_center_x)
        center_y_px = int(height * self.config.zoom_center_y)
        
        # Calculate ROI boundaries
        x1 = max(0, center_x_px - new_width // 2)
        y1 = max(0, center_y_px - new_height // 2)
        x2 = min(width, x1 + new_width)
        y2 = min(height, y1 + new_height)
        
        # Adjust if ROI goes out of bounds
        if x2 - x1 < new_width:
            if x1 == 0:
                x2 = min(width, new_width)
            else:
                x1 = max(0, width - new_width)
        
        if y2 - y1 < new_height:
            if y1 == 0:
                y2 = min(height, new_height)
            else:
                y1 = max(0, height - new_height)
        
        return (x1, y1, x2, y2)
    
    def apply_zoom(self, frame):
        """Apply zoom to the frame."""
        if not self.config.zoom_enabled or self.config.zoom_factor <= 1.0:
            return frame, None
        
        # Calculate ROI if not cached or frame size changed
        if self.zoom_roi is None or self.zoom_roi[2] - self.zoom_roi[0] != int(frame.shape[1] / self.config.zoom_factor):
            self.zoom_roi = self.calculate_zoom_roi(frame.shape)
        
        x1, y1, x2, y2 = self.zoom_roi
        
        # Extract ROI
        zoomed_frame = frame[y1:y2, x1:x2]
        
        # Resize back to original dimensions
        zoomed_frame = cv2.resize(zoomed_frame, (frame.shape[1], frame.shape[0]), 
                                 interpolation=cv2.INTER_LINEAR)
        
        return zoomed_frame, self.zoom_roi
    
    def map_coordinates_to_original(self, x, y, zoom_roi, original_frame_shape):
        """Map coordinates from zoomed image back to original image."""
        if zoom_roi is None:
            return x, y
        
        x1, y1, x2, y2 = zoom_roi
        roi_width = x2 - x1
        roi_height = y2 - y1
        
        # Map from zoomed (resized) coordinates to original
        original_x = int(x * roi_width / original_frame_shape[1] + x1)
        original_y = int(y * roi_height / original_frame_shape[0] + y1)
        
        return original_x, original_y
