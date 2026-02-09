#!/usr/bin/env python3

import time
from collections import deque


class DetectionFilters:
    """Handles temporal consistency and hysteresis filtering."""
    
    def __init__(self, config, logger):
        """
        Initialize detection filters.
        
        Args:
            config: Configuration object
            logger: ROS2 logger
        """
        self.config = config
        self.logger = logger
        
        # Temporal consistency tracking
        self.detection_history = deque(maxlen=config.temporal_window_size)
        self.temporal_filter_ready = False
        
        # Hysteresis state tracking
        self.currently_in_detection = False
        self.detection_start_time = None
    
    def apply_temporal_filter(self, current_frame_detection):
        """
        Apply temporal consistency check to reduce false positives.
        
        Args:
            current_frame_detection (bool): Detection result for current frame
            
        Returns:
            bool: Final detection decision after temporal filtering
        """
        # Add current frame detection to history
        self.detection_history.append(current_frame_detection)
        
        # Check if we have enough frames for temporal filtering
        if len(self.detection_history) < self.config.temporal_window_size:
            # Not enough data yet, default to no detection during startup
            if not self.temporal_filter_ready and len(self.detection_history) == self.config.temporal_window_size:
                self.temporal_filter_ready = True
                self.logger.info('✔ Temporal filter ready! Detection system now active.')
            return False
        
        # Count how many recent frames detected a crack
        detection_count = sum(self.detection_history)
        
        # Apply temporal threshold
        is_crack_detected = detection_count >= self.config.temporal_threshold
        
        return is_crack_detected
    
    def apply_hysteresis(self, crack_percentage, temporal_detection):
        """
        Apply hysteresis to prevent rapid on/off flickering.
        
        Args:
            crack_percentage (float): Current frame's crack percentage
            temporal_detection (bool): Result from temporal filter
            
        Returns:
            bool: Final detection decision with hysteresis applied
        """
        if not self.config.hysteresis_enabled:
            # Hysteresis disabled, just use temporal detection
            return temporal_detection
        
        current_time = time.time()
        
        if not self.currently_in_detection:
            # Currently NOT detecting
            # Turn ON if temporal filter says crack AND percentage above upper threshold
            if temporal_detection and crack_percentage >= self.config.min_crack_percent:
                self.currently_in_detection = True
                self.detection_start_time = current_time
                self.logger.info(f'🔵 Detection ACTIVATED (crack: {crack_percentage:.2f}%)')
                return True
            else:
                return False
        
        else:
            # Currently IN detection mode
            
            # Check minimum duration (time-based hysteresis)
            time_in_detection = current_time - self.detection_start_time
            if time_in_detection < self.config.hysteresis_min_duration:
                # Still within minimum duration, stay ON
                return True
            
            # Check if we should turn OFF
            # Turn OFF only if percentage drops below LOWER threshold
            if crack_percentage < self.config.hysteresis_low_threshold:
                self.currently_in_detection = False
                self.detection_start_time = None
                self.logger.info(f'🔵 Detection DEACTIVATED (crack: {crack_percentage:.2f}%)')
                return False
            else:
                # Stay ON (above lower threshold)
                return True
    
    def validate_crack_geometry(self, crack_info):
        """
        Validate a crack blob based on geometric properties.
        Filters out shadows (too round) and wires (too thin).
        
        Args:
            crack_info (dict): Dictionary with 'area', 'aspect_ratio', 'bbox' keys
            
        Returns:
            bool: True if crack passes geometric validation, False otherwise
        """
        # Check area threshold
        if crack_info['area'] < self.config.min_crack_area:
            return False
        
        # Check aspect ratio
        aspect = crack_info['aspect_ratio']
        if aspect < self.config.min_aspect_ratio or aspect > self.config.max_aspect_ratio:
            return False
        
        # All checks passed
        return True
    
    def validate_crack_list(self, crack_list):
        """
        Apply geometric validation to a list of crack candidates.
        
        Args:
            crack_list (list): List of crack dictionaries from ScanPointExtractor
            
        Returns:
            list: Filtered list containing only geometrically valid cracks
        """
        valid_cracks = []
        
        for crack in crack_list:
            if self.validate_crack_geometry(crack):
                valid_cracks.append(crack)
            else:
                # Log rejection for debugging
                self.logger.debug(
                    f"Crack ID {crack['id']} rejected: "
                    f"area={crack['area']}px, aspect_ratio={crack['aspect_ratio']:.2f}"
                )
        
        return valid_cracks
