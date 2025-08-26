"""
SAM2 Utilities for Density Segmentation GUI
"""
import torch
import numpy as np
import cv2
import streamlit as st
import os

# SAM2 imports with fallback
try:
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    build_sam2 = None
    build_sam2_video_predictor = None
    SAM2ImagePredictor = None
    SAM2VideoPredictor = None

# Model configuration
CHECKPOINT_PATH = "checkpoints/sam2.1_hiera_large.pt"

class SAM2Manager:
    """Manager class for SAM2 model operations"""
    
    def __init__(self):
        self.predictor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
    
    def check_dependencies(self):
        """Check if SAM2 dependencies are available"""
        if not SAM2_AVAILABLE:
            return False, "SAM2 not installed. Run: pip install git+https://github.com/facebookresearch/segment-anything-2.git"
        
        if not torch.cuda.is_available():
            st.warning("CUDA unavailable. SAM2 will run on CPU (not recommended).")
        
        return True, "Dependencies OK"
    
    def check_checkpoint(self):
        """Check if SAM2 checkpoint file exists"""
        if not os.path.exists(CHECKPOINT_PATH):
            return False, f"Checkpoint not found: {CHECKPOINT_PATH}"
        return True, "Checkpoint found"
    
    def load_model(self):
        """Load the SAM2 model"""
        if self.model_loaded:
            return True, "Model already loaded"
        
        # Check dependencies
        deps_ok, deps_msg = self.check_dependencies()
        if not deps_ok:
            return False, deps_msg

        # Check checkpoint
        checkpoint_ok, checkpoint_msg = self.check_checkpoint()
        if not checkpoint_ok:
            return False, checkpoint_msg
        
        try:
            # Try different config approaches for SAM2
            # Based on the actual config structure found in your installation
            config_options = [
                "sam2.1_hiera_l.yaml",                 # Direct config name
                "configs/sam2.1_hiera_l.yaml",         # With configs/ prefix
                "sam2.1/sam2.1_hiera_l.yaml",          # From sam2.1 folder
                "sam2_hiera_l.yaml",                   # Alternative name
                "sam2.1_hiera_large.yaml",             # Another alternative
            ]
            
            sam2_model = None
            config_used = None
            last_error = None
            
            for config_name in config_options:
                try:
                    sam2_model = build_sam2(config_name, CHECKPOINT_PATH, device=self.device)
                    config_used = config_name
                    break
                except Exception as config_error:
                    last_error = str(config_error)
                    continue
            
            if sam2_model is None:
                # If all specific configs fail, let's try to find the actual config file
                try:
                    if SAM2_AVAILABLE:
                        import sam2
                        sam2_path = os.path.dirname(sam2.__file__)
                        
                        # Common config locations to try
                        possible_configs = [
                            os.path.join(sam2_path, "configs", "sam2.1_hiera_l.yaml"),
                            os.path.join(sam2_path, "sam2.1", "sam2.1_hiera_l.yaml"), 
                            os.path.join(sam2_path, "configs", "sam2.1", "sam2.1_hiera_l.yaml"),
                        ]
                        
                        for config_path in possible_configs:
                            if os.path.exists(config_path):
                                sam2_model = build_sam2(config_path, CHECKPOINT_PATH, device=self.device)
                                config_used = f"found at: {config_path}"
                                break
                        
                        if sam2_model is None:
                            # Final fallback: try to build without config (auto-detect)
                            sam2_model = build_sam2(None, CHECKPOINT_PATH, device=self.device)
                            config_used = "auto-detected"
                            
                except Exception as final_error:
                    error_msg = f"Error loading model with all configs. Last error: {last_error}"
                    error_msg += f"\nFinal error: {str(final_error)}"
                    error_msg += "\n\nTroubleshooting:\n"
                    error_msg += "1. Make sure SAM2 is properly installed: pip install git+https://github.com/facebookresearch/segment-anything-2.git\n"
                    error_msg += "2. Check if the checkpoint file exists at: " + CHECKPOINT_PATH + "\n"
                    error_msg += "3. Try reinstalling SAM2 and PyTorch with CUDA support"
                    return False, error_msg
            
            self.predictor = SAM2ImagePredictor(sam2_model)
            self.model_loaded = True
            return True, f"SAM2 model loaded successfully on device: {self.device} using config: {config_used}"
            
        except Exception as e:
            return False, f"Unexpected error loading model: {str(e)}"
    
    def load_video_predictor(self):
        """Load SAM2 video predictor for advanced pipeline"""
        if not SAM2_AVAILABLE:
            return False, "SAM2 not available"
        
        # Check checkpoint
        checkpoint_ok, checkpoint_msg = self.check_checkpoint()
        if not checkpoint_ok:
            return False, checkpoint_msg
        
        try:
            # Try different config approaches for SAM2 video predictor
            config_options = [
                "sam2.1_hiera_l.yaml",
                "configs/sam2.1_hiera_l.yaml",
                "sam2.1/sam2.1_hiera_l.yaml",
                "sam2_hiera_l.yaml",
                "sam2.1_hiera_large.yaml",
            ]
            
            video_predictor = None
            config_used = None
            last_error = None
            
            for config_name in config_options:
                try:
                    video_predictor = build_sam2_video_predictor(config_name, CHECKPOINT_PATH, device=self.device)
                    config_used = config_name
                    break
                except Exception as config_error:
                    last_error = str(config_error)
                    continue
            
            if video_predictor is None:
                # Try to find config file in SAM2 installation
                try:
                    if SAM2_AVAILABLE:
                        import sam2
                        sam2_path = os.path.dirname(sam2.__file__)
                        
                        possible_configs = [
                            os.path.join(sam2_path, "configs", "sam2.1_hiera_l.yaml"),
                            os.path.join(sam2_path, "sam2.1", "sam2.1_hiera_l.yaml"), 
                            os.path.join(sam2_path, "configs", "sam2.1", "sam2.1_hiera_l.yaml"),
                        ]
                        
                        for config_path in possible_configs:
                            if os.path.exists(config_path):
                                video_predictor = build_sam2_video_predictor(config_path, CHECKPOINT_PATH, device=self.device)
                                config_used = f"found at: {config_path}"
                                break
                                
                except Exception as final_error:
                    error_msg = f"Error loading video predictor. Last error: {last_error}"
                    error_msg += f"\nFinal error: {str(final_error)}"
                    return False, error_msg
            
            if video_predictor is None:
                return False, f"Failed to load video predictor with all configs. Last error: {last_error}"
                
            return True, video_predictor, f"SAM2 video predictor loaded successfully using config: {config_used}"
            
        except Exception as e:
            return False, f"Unexpected error loading video predictor: {str(e)}"
    
    