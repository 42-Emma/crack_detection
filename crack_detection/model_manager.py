#!/usr/bin/env python3

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from crack_detection.unet_model import UNet
from crack_detection.train_pix2pix import Generator
from crack_detection.unet_rt_inference_tiling6 import TiledUNetInference
from crack_detection.pix2pix_rt_inference_tiling import TiledRefinedInference


class ModelManager:
    """Manages model loading and initialization."""
    
    def __init__(self, config, device, logger):
        """
        Initialize model manager.
        
        Args:
            config: Configuration object
            device: torch device (cuda/cpu)
            logger: ROS2 logger
        """
        self.config = config
        self.device = device
        self.logger = logger
        
        self.mode = None
        self.model = None
        self.pix2pix = None
        self.transforms = None
        self.tiled_inferencer = None
        self.tiled_refined_inferencer = None
        
        # Determine mode and initialize models
        self.determine_and_initialize_mode()
    
    def determine_and_initialize_mode(self):
        """Determine which inference mode to use and initialize models."""
        
        if self.config.use_tiling and self.config.use_pix2pix:
            # Mode 4: UNet + Pix2Pix Tiled (Best Quality)
            self.mode = "UNET_PIX2PIX_TILED"
            self.logger.info(f'Mode: UNet + Pix2Pix TILED (Best Quality)')
            self.logger.info(f'Window size: {self.config.window_size}x{self.config.window_size}')
            self.logger.info(f'Subdivisions: {self.config.subdivisions}')
            
            self.tiled_refined_inferencer = TiledRefinedInference(
                unet_path=self.config.model_path,
                pix2pix_path=self.config.pix2pix_model_path,
                device=self.device,
                window_size=self.config.window_size,
                subdivisions=self.config.subdivisions
            )
            self.model = None
            self.pix2pix = None
            self.transforms = None
            self.tiled_inferencer = None
            
        elif self.config.use_tiling and not self.config.use_pix2pix:
            # Mode 2: UNet Tiled Only
            self.mode = "UNET_TILED"
            self.logger.info(f'Mode: UNet TILED (High Accuracy)')
            self.logger.info(f'Window size: {self.config.window_size}x{self.config.window_size}')
            self.logger.info(f'Subdivisions: {self.config.subdivisions}')
            
            self.tiled_inferencer = TiledUNetInference(
                model_path=self.config.model_path,
                device=self.device,
                window_size=self.config.window_size,
                subdivisions=self.config.subdivisions
            )
            self.model = None
            self.pix2pix = None
            self.transforms = None
            self.tiled_refined_inferencer = None
            
        elif not self.config.use_tiling and self.config.use_pix2pix:
            # Mode 3: UNet + Pix2Pix Fast
            self.mode = "UNET_PIX2PIX_FAST"
            self.logger.info(f'Mode: UNet + Pix2Pix FAST (Better Quality)')
            self.logger.info(f'Input size: {self.config.input_size}x{self.config.input_size}')
            
            self.model, self.pix2pix = self.load_both_models()
            self.transforms = self.get_inference_transforms()
            self.tiled_inferencer = None
            self.tiled_refined_inferencer = None
            
        else:
            # Mode 1: UNet Fast Only (Default)
            self.mode = "UNET_FAST"
            self.logger.info(f'Mode: UNet FAST (Real-time)')
            self.logger.info(f'Input size: {self.config.input_size}x{self.config.input_size}')
            
            self.model = self.load_unet_model()
            self.pix2pix = None
            self.transforms = self.get_inference_transforms()
            self.tiled_inferencer = None
            self.tiled_refined_inferencer = None
    
    def load_unet_model(self):
        """Load UNet model only."""
        self.logger.info(f'Loading UNet from: {self.config.model_path}')
        
        model = UNet(
            num_classes=1,
            align_corners=False,
            use_deconv=False,
            in_channels=3
        ).to(self.device)
        
        checkpoint = torch.load(self.config.model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        self.logger.info('✓ UNet loaded successfully!')
        return model
    
    def load_both_models(self):
        """Load both UNet and Pix2Pix models."""
        self.logger.info(f'Loading UNet from: {self.config.model_path}')
        
        # Load UNet
        unet = UNet(
            num_classes=1,
            align_corners=False,
            use_deconv=False,
            in_channels=3
        ).to(self.device)
        
        unet_checkpoint = torch.load(self.config.model_path, map_location=self.device, weights_only=False)
        unet.load_state_dict(unet_checkpoint['model_state_dict'])
        unet.eval()
        
        self.logger.info(f'Loading Pix2Pix from: {self.config.pix2pix_model_path}')
        
        # Load Pix2Pix
        pix2pix = Generator().to(self.device)
        
        pix2pix_checkpoint = torch.load(self.config.pix2pix_model_path, map_location=self.device, weights_only=False)
        pix2pix.load_state_dict(pix2pix_checkpoint['generator_state_dict'])
        pix2pix.eval()
        
        self.logger.info('✓ Both models loaded successfully!')
        return unet, pix2pix
    
    def get_inference_transforms(self):
        """Get transforms for inference."""
        return A.Compose([
            A.Resize(self.config.input_size, self.config.input_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
