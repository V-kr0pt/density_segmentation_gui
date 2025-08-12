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
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

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
            # Use config name that SAM2 finds automatically
            config_name = "sam2.1_hiera_l.yaml"

            # Load model
            sam2_model = build_sam2(config_name, CHECKPOINT_PATH, device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            self.model_loaded = True
            return True, f"SAM2 model loaded successfully on device: {self.device}"
            
        except Exception as e:
            return False, f"Error loading model: {str(e)}"
    
    def set_image(self, image):
        """Set the image for segmentation"""
        if not self.model_loaded:
            return False, "Model not loaded"
        
        try:
            # Convert to RGB format if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            self.predictor.set_image(image_rgb)
            return True, "Image loaded successfully"
            
        except Exception as e:
            return False, f"Error loading image: {str(e)}"
    
    def predict(self, input_points=None, input_labels=None, input_boxes=None):
        """Perform SAM2 prediction"""
        if not self.model_loaded:
            return None, None, "Model not loaded"
        
        try:
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=input_boxes,
                multimask_output=True
            )
            
            return masks, scores, "Prediction completed successfully"
            
        except Exception as e:
            return None, None, f"Prediction error: {str(e)}"

def show_sam2_setup_info():
    """Display SAM2 setup information"""
    st.info("""
    **SAM2 Setup Required:**
    
    1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    or
    
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install git+https://github.com/facebookresearch/segment-anything-2.git
    pip install supervision
    ```
    """)

def convert_nii_slice_for_sam2(nii_slice):
    """Convert a NII slice to SAM2 compatible format"""
    # Normalize to 0-255
    normalized = ((nii_slice - nii_slice.min()) / (nii_slice.max() - nii_slice.min()) * 255).astype(np.uint8)
    
    # Convert to RGB (SAM2 expects RGB images)
    if len(normalized.shape) == 2:
        rgb_image = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    else:
        rgb_image = normalized
    
    return rgb_image
