#!/usr/bin/env python3

import torch
import numpy as np
import gc


class InferenceEngine:
    """Handles all model inference operations."""
    
    def __init__(self, model_manager, config, device):
        """
        Initialize inference engine.
        
        Args:
            model_manager: ModelManager instance
            config: Configuration object
            device: torch device
        """
        self.model_manager = model_manager
        self.config = config
        self.device = device
    
    def predict_frame(self, frame_rgb):
        """Route to appropriate prediction method based on mode."""
        if self.model_manager.mode == "UNET_FAST":
            return self.predict_frame_unet_fast(frame_rgb)
        elif self.model_manager.mode == "UNET_TILED":
            return self.predict_frame_unet_tiled(frame_rgb)
        elif self.model_manager.mode == "UNET_PIX2PIX_FAST":
            return self.predict_frame_pix2pix_fast(frame_rgb)
        elif self.model_manager.mode == "UNET_PIX2PIX_TILED":
            return self.predict_frame_pix2pix_tiled(frame_rgb)
    
    def predict_frame_unet_fast(self, frame_rgb):
        """Mode 1: Fast UNet prediction."""
        original_size = frame_rgb.shape[:2]
        
        # Apply transforms
        transformed = self.model_manager.transforms(image=frame_rgb)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model_manager.model(input_tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            
            prediction = torch.sigmoid(output)
            
            # Resize back to original size
            pred_resized = torch.nn.functional.interpolate(
                prediction,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )
            
            # Convert to numpy
            pred_numpy = pred_resized.squeeze().cpu().numpy()
            
            # Apply threshold
            binary_mask = (pred_numpy > self.config.threshold).astype(np.uint8) * 255
        
        # Calculate crack percentage
        crack_pixels = np.sum(binary_mask > 127)
        total_pixels = binary_mask.size
        crack_percentage = (crack_pixels / total_pixels) * 100
        
        return pred_numpy, binary_mask, crack_percentage
    
    def predict_frame_unet_tiled(self, frame_rgb):
        """Mode 2: Tiled UNet prediction."""
        original_size = frame_rgb.shape[:2]
        
        # Pad image
        padded = self.model_manager.tiled_inferencer._pad_img(frame_rgb)
        padded_shape = list(padded.shape[:-1]) + [1]
        
        # Process with tiling
        subdivs = self.model_manager.tiled_inferencer._windowed_subdivs(padded)
        predictions = self.model_manager.tiled_inferencer._recreate_from_subdivs(subdivs, padded_shape)
        
        # Unpad
        predictions = self.model_manager.tiled_inferencer._unpad_img(predictions)
        
        # Crop to original size
        pred_numpy = predictions[:original_size[0], :original_size[1], 0]
        
        # Apply threshold
        binary_mask = (pred_numpy > self.config.threshold).astype(np.uint8) * 255
        
        # Calculate crack percentage
        crack_pixels = np.sum(binary_mask > 127)
        total_pixels = binary_mask.size
        crack_percentage = (crack_pixels / total_pixels) * 100
        
        # Cleanup
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return pred_numpy, binary_mask, crack_percentage
    
    def predict_frame_pix2pix_fast(self, frame_rgb):
        """Mode 3: Fast UNet + Pix2Pix prediction."""
        original_size = frame_rgb.shape[:2]
        
        # Apply transforms
        transformed = self.model_manager.transforms(image=frame_rgb)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Stage 1: UNet prediction
            unet_output = self.model_manager.model(input_tensor)
            if isinstance(unet_output, (tuple, list)):
                unet_output = unet_output[0]
            unet_pred = torch.sigmoid(unet_output)
            
            # Stage 2: Pix2Pix residual refinement
            residual = self.model_manager.pix2pix(unet_pred)
            
            # Stage 3: Combine predictions
            refined_pred = torch.clamp(unet_pred + residual, 0, 1)
            
            # Resize back to original size
            pred_resized = torch.nn.functional.interpolate(
                refined_pred,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )
            
            # Convert to numpy
            pred_numpy = pred_resized.squeeze().cpu().numpy()
            
            # Apply threshold
            binary_mask = (pred_numpy > self.config.threshold).astype(np.uint8) * 255
        
        # Calculate crack percentage
        crack_pixels = np.sum(binary_mask > 127)
        total_pixels = binary_mask.size
        crack_percentage = (crack_pixels / total_pixels) * 100
        
        return pred_numpy, binary_mask, crack_percentage
    
    def predict_frame_pix2pix_tiled(self, frame_rgb):
        """Mode 4: Tiled UNet + Pix2Pix prediction."""
        original_size = frame_rgb.shape[:2]
        
        # Pad image
        padded = self.model_manager.tiled_refined_inferencer._pad_img(frame_rgb)
        padded_shape = list(padded.shape[:-1]) + [1]
        
        # Process with tiling + Pix2Pix
        subdivs = self.model_manager.tiled_refined_inferencer._windowed_subdivs(padded)
        predictions = self.model_manager.tiled_refined_inferencer._recreate_from_subdivs(subdivs, padded_shape)
        
        # Unpad
        predictions = self.model_manager.tiled_refined_inferencer._unpad_img(predictions)
        
        # Crop to original size
        pred_numpy = predictions[:original_size[0], :original_size[1], 0]
        
        # Apply threshold
        binary_mask = (pred_numpy > self.config.threshold).astype(np.uint8) * 255
        
        # Calculate crack percentage
        crack_pixels = np.sum(binary_mask > 127)
        total_pixels = binary_mask.size
        crack_percentage = (crack_pixels / total_pixels) * 100
        
        # Cleanup
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return pred_numpy, binary_mask, crack_percentage
