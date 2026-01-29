#!/usr/bin/env python3

import cv2
import numpy as np
import time
from collections import deque


class Visualization:
    """Handles visualization creation."""
    
    def __init__(self, config):
        """
        Initialize visualization.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # FPS tracking
        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()
    
    def calculate_fps(self):
        """Calculate current FPS."""
        current_time = time.time()
        time_diff = current_time - self.last_time
        self.last_time = current_time
        
        if time_diff > 0:
            fps = 1.0 / time_diff
            self.fps_queue.append(fps)
        
        return np.mean(self.fps_queue) if len(self.fps_queue) > 0 else 0.0
    
    def create_visualization(self, frame_bgr, pred_prob, binary_mask, crack_percentage, 
                           is_crack_detected, zoom_roi, temporal_status, depth_viz, 
                           wall_distance, last_original_frame, mode, frame_count, 
                           currently_in_detection):
        """Create visualization with heatmap and overlay."""
        # Store original frame for zoom visualization
        original_frame = last_original_frame.copy() if last_original_frame is not None else frame_bgr.copy()
        
        # Draw zoom region on original frame if zoom is enabled
        if self.config.zoom_enabled and zoom_roi is not None:
            x1, y1, x2, y2 = zoom_roi
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(original_frame, f"Zoom: {self.config.zoom_factor:.1f}x", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Create heatmap
        heatmap = (pred_prob * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Create crack overlay
        overlay = frame_bgr.copy()
        crack_pixels = binary_mask > 127
        overlay[crack_pixels] = [0, 0, 255]  # Red for cracks
        blended = cv2.addWeighted(frame_bgr, 0.7, overlay, 0.3, 0)
        
        
        # Create depth visualization if available
        if depth_viz is not None:
            # Get target dimensions from RGB frame
            target_height = original_frame.shape[0]
            target_width = original_frame.shape[1]
            
            # Resize depth mask to match
            depth_viz_resized = cv2.resize(depth_viz,
                                        (target_width, target_height),
                                        interpolation=cv2.INTER_NEAREST)
            
            # Create colored depth visualization
            depth_colored = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # Overlay valid depth region in green
            depth_colored[depth_viz_resized > 0] = [0, 255, 0]
            
            # Stack: original, heatmap, blended, depth (all same size now)
            combined = np.hstack([original_frame, heatmap_colored, blended, depth_colored])
        else:
            # Stack without depth: original, heatmap, blended
            combined = np.hstack([original_frame, heatmap_colored, blended])
        
        # Add info panel
        panel_height = 120
        info_panel = np.zeros((panel_height, combined.shape[1], 3), dtype=np.uint8)
        
        # FPS
        fps = self.calculate_fps()
        cv2.putText(info_panel, f"FPS: {fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Crack percentage
        cv2.putText(info_panel, f"Crack: {crack_percentage:.2f}%", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Wall distance
        if wall_distance is not None:
            cv2.putText(info_panel, f"Wall: {wall_distance:.2f}m", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Zoom status
        zoom_text = f"Zoom: {self.config.zoom_factor:.1f}x" if self.config.zoom_enabled else "Zoom: OFF"
        cv2.putText(info_panel, zoom_text, (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Temporal filter status
        cv2.putText(info_panel, f"Temporal: {temporal_status}", (250, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Hysteresis status
        hysteresis_text = "ON" if currently_in_detection else "OFF"
        hysteresis_color = (0, 255, 0) if currently_in_detection else (100, 100, 100)
        cv2.putText(info_panel, f"Hysteresis: {hysteresis_text}", (250, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, hysteresis_color, 1)
        
        # Depth filter status
        depth_text = "ADAPTIVE" if self.config.use_adaptive_depth else f"FIXED {self.config.target_inspection_distance}m"
        if self.config.depth_filtering_enabled:
            cv2.putText(info_panel, f"Depth: {depth_text}", (250, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Mode
        cv2.putText(info_panel, f"Mode: {mode}", (250, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Frame count
        cv2.putText(info_panel, f"Frame: {frame_count}", (500, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Capture instruction
        cv2.putText(info_panel, "Publish Bool to /capture_image", (500, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
        
        # Detection status
        if is_crack_detected:
            status_text = "CRACK DETECTED"
            status_color = (0, 0, 255)  # Red
        else:
            status_text = "NO CRACK"
            status_color = (0, 255, 0)  # Green
        
        text_x = combined.shape[1] - 250
        cv2.putText(info_panel, status_text, (text_x, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Combine with info panel
        result = np.vstack([info_panel, combined])
        
        return result
        
    def create_scan_points_visualization(self, frame_bgr, binary_mask, scan_points):
        """
        Create visualization showing only crack overlay with scan points and values.
        
        Args:
            frame_bgr: Original BGR image
            binary_mask: Binary crack mask
            scan_points: Dictionary with 'center', 'left_point', 'right_point'
            
        Returns:
            Visualization image with crack overlay and scan points
        """
        # Create crack overlay
        result = frame_bgr.copy()
        crack_pixels = binary_mask > 127
        overlay = result.copy()
        overlay[crack_pixels] = [0, 0, 255]  # Red for cracks
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
        
        if scan_points is None:
            return result
        
        # Extract points
        center_x, center_y = scan_points['center']
        left_x, left_y = scan_points['left_point']
        right_x, right_y = scan_points['right_point']
        
        # Draw points with larger, more visible circles
        cv2.circle(result, (center_x, center_y), 12, (255, 0, 255), -1)  # Purple center
        cv2.circle(result, (center_x, center_y), 14, (255, 255, 255), 2)  # White outline
        
        cv2.circle(result, (left_x, left_y), 12, (0, 255, 255), -1)      # Yellow left/start
        cv2.circle(result, (left_x, left_y), 14, (255, 255, 255), 2)      # White outline
        
        cv2.circle(result, (right_x, right_y), 12, (0, 255, 255), -1)    # Yellow right/end
        cv2.circle(result, (right_x, right_y), 14, (255, 255, 255), 2)    # White outline
        
        # Add labels with coordinates
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Center label and coordinates
        center_label = f"CENTER"
        center_coords = f"({center_x}, {center_y})"
        cv2.putText(result, center_label, (center_x + 20, center_y - 25),
                   font, font_scale, (255, 0, 255), thickness)
        cv2.putText(result, center_coords, (center_x + 20, center_y - 5),
                   font, font_scale * 0.8, (255, 0, 255), thickness)
        
        # Start label and coordinates
        start_label = f"START"
        start_coords = f"({left_x}, {left_y})"
        cv2.putText(result, start_label, (left_x + 20, left_y - 25),
                   font, font_scale, (0, 255, 255), thickness)
        cv2.putText(result, start_coords, (left_x + 20, left_y - 5),
                   font, font_scale * 0.8, (0, 255, 255), thickness)
        
        # End label and coordinates
        end_label = f"END"
        end_coords = f"({right_x}, {right_y})"
        cv2.putText(result, end_label, (right_x + 20, right_y - 25),
                   font, font_scale, (0, 255, 255), thickness)
        cv2.putText(result, end_coords, (right_x + 20, right_y - 5),
                   font, font_scale * 0.8, (0, 255, 255), thickness)
        
        # Add title at top
        title = "Crack Scan Points"
        cv2.rectangle(result, (0, 0), (result.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(result, title, (10, 35),
                   font, 1.0, (0, 255, 0), 2)
        
        return result
