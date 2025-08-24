import os
import numpy as np
import streamlit as st
import nibabel as nib
from PIL import Image
import shutil
import cv2
import torch
from utils import ImageOperations, MaskOperations, ThresholdOperations  # Added MaskOperations
from sam_utils import SAM2Manager
from sam2_simple_approach import (
    create_simple_initial_mask, 
    safe_roi_extraction, 
    safe_normalize_for_sam2, 
    apply_safe_threshold
)

# SAM2 Video Predictor imports with fallback
try:
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    SAM2_VIDEO_AVAILABLE = True
except ImportError:
    SAM2_VIDEO_AVAILABLE = False

def test_sam2_installation():
    """Test SAM2 installation and display diagnostic info"""
    print("=== SAM2 Installation Diagnostic ===")
    
    # Check SAM2 import
    try:
        import sam2
        print(f"✅ SAM2 imported successfully")
        print(f"   SAM2 path: {sam2.__file__}")
        
        # Check version if available
        if hasattr(sam2, '__version__'):
            print(f"   SAM2 version: {sam2.__version__}")
    except ImportError as e:
        print(f"❌ SAM2 import failed: {e}")
        return False
    
    # Check required imports
    try:
        from sam2.build_sam import build_sam2_video_predictor
        print(f"✅ build_sam2_video_predictor imported")
    except ImportError as e:
        print(f"❌ build_sam2_video_predictor import failed: {e}")
        return False
    
    # Check configs
    configs = find_sam2_configs()
    if configs:
        print(f"✅ Found {len(configs)} config files:")
        for name, rel_path, full_path in configs[:5]:  # Show first 5
            print(f"   {name}")
    else:
        print(f"❌ No config files found")
        return False
    
    # Check checkpoint
    checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
    if os.path.exists(checkpoint_path):
        file_size = os.path.getsize(checkpoint_path) / (1024**3)  # GB
        print(f"✅ Checkpoint found: {checkpoint_path} ({file_size:.2f} GB)")
    else:
        print(f"❌ Checkpoint missing: {checkpoint_path}")
        return False
    
    # Check PyTorch and CUDA
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
    
    return True

def extract_roi_traditional_approach(nii_data, mask_data, middle_image_slice, middle_mask_slice, num_slices, padding=10):
    """
    Extract ROI using traditional approach like the working mode
    
    Args:
        nii_data: 3D NIfTI data (slices, height, width)
        mask_data: 3D mask data (slices, height, width)
        middle_image_slice: Middle slice for ROI calculation (height, width)
        middle_mask_slice: Middle mask slice (height, width)
        num_slices: Number of slices
        padding: Padding around ROI
    
    Returns:
        roi_bounds, roi_slices, roi_masks, success
    """
    try:
        print(f"Traditional ROI extraction for {num_slices} slices")
        
        # Find ROI bounds from the middle mask slice (like traditional mode)
        y_coords, x_coords = np.where(middle_mask_slice > 0)
        
        if len(y_coords) == 0 or len(x_coords) == 0:
            print("No mask data found in middle slice")
            return None, None, None, False
        
        # Calculate bounds with padding
        y_min = max(0, np.min(y_coords) - padding)
        y_max = min(middle_image_slice.shape[0], np.max(y_coords) + padding)
        x_min = max(0, np.min(x_coords) - padding)
        x_max = min(middle_image_slice.shape[1], np.max(x_coords) + padding)
        
        print(f"ROI bounds: y({y_min},{y_max}) x({x_min},{x_max})")
        
        # Validate bounds
        if y_min >= y_max or x_min >= x_max:
            print("Invalid ROI bounds")
            return None, None, None, False
        
        roi_bounds = (x_min, y_min, x_max, y_max)
        
        # Extract ROI from all slices (following traditional slice-by-slice approach)
        roi_slices = []
        roi_masks = []
        
        for slice_idx in range(num_slices):
            # Extract slice like traditional mode: nii_data[slice_idx, :, :]
            image_slice = nii_data[slice_idx, :, :]  # Shape: (height, width)
            
            # Extract corresponding mask slice
            if slice_idx < mask_data.shape[0]:
                mask_slice = mask_data[slice_idx, :, :]
            else:
                # Use last available mask slice if fewer mask slices than image slices
                mask_slice = mask_data[-1, :, :]
            
            # Extract ROI from both image and mask
            roi_image = image_slice[y_min:y_max, x_min:x_max]
            roi_mask = mask_slice[y_min:y_max, x_min:x_max]
            
            roi_slices.append(roi_image)
            roi_masks.append(roi_mask)
        
        print(f"Successfully extracted ROI from {len(roi_slices)} slices")
        print(f"ROI size: {roi_slices[0].shape if roi_slices else 'None'}")
        
        return roi_bounds, roi_slices, roi_masks, True
        
    except Exception as e:
        print(f"Error in traditional ROI extraction: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, False

def create_temp_jpeg_folder(processed_slices, temp_dir_base="temp_sam2_frames"):
    """
    Create a temporary directory with JPEG files for SAM2 video processing
    
    Args:
        processed_slices: List of 2D numpy arrays (grayscale images)
        temp_dir_base: Base name for temporary directory
    
    Returns:
        temp_dir_path: Path to the created directory
        frame_names: List of frame filenames
        success: Boolean indicating success
    """
    try:
        import tempfile
        import shutil
        from PIL import Image
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=temp_dir_base)
        frame_names = []
        
        for i, slice_data in enumerate(processed_slices):
            # Ensure slice is 2D
            if len(slice_data.shape) == 3:
                slice_data = slice_data[:, :, 0]  # Take first channel
            
            # Normalize to 0-255 range for JPEG
            if slice_data.dtype != np.uint8:
                slice_min = np.min(slice_data)
                slice_max = np.max(slice_data)
                if slice_max > slice_min:
                    slice_normalized = ((slice_data - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
                else:
                    slice_normalized = np.zeros_like(slice_data, dtype=np.uint8)
            else:
                slice_normalized = slice_data
            
            # Convert to RGB for SAM2 (SAM2 expects 3-channel images)
            slice_rgb = np.stack([slice_normalized] * 3, axis=-1)
            
            # Create filename with zero-padding for proper sorting
            frame_filename = f"{i:05d}.jpg"  # e.g., "00000.jpg", "00001.jpg"
            frame_path = os.path.join(temp_dir, frame_filename)
            
            # Save as JPEG
            pil_image = Image.fromarray(slice_rgb)
            pil_image.save(frame_path, "JPEG", quality=95)
            
            frame_names.append(frame_filename)
        
        print(f"Created temporary JPEG folder: {temp_dir}")
        print(f"Generated {len(frame_names)} frame files")
        
        # Show sample frames for visualization
        if len(processed_slices) > 0:
            print(f"Sample frame info:")
            print(f"  First frame shape: {processed_slices[0].shape}")
            print(f"  First frame data type: {processed_slices[0].dtype}")
            print(f"  First frame value range: [{np.min(processed_slices[0]):.2f}, {np.max(processed_slices[0]):.2f}]")
        
        return temp_dir, frame_names, True
        
    except Exception as e:
        print(f"Error creating temporary JPEG folder: {e}")
        return None, None, False

def create_sam2_progress_visualization(roi_slices, roi_masks, roi_bounds, middle_slice_idx=None):
    """Create visualization to show SAM2 processing progress"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        if not roi_slices or len(roi_slices) == 0:
            return None
        
        # Use middle slice if not specified
        if middle_slice_idx is None:
            middle_slice_idx = len(roi_slices) // 2
        
        if middle_slice_idx >= len(roi_slices):
            middle_slice_idx = 0
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Show original ROI slice
        roi_slice = roi_slices[middle_slice_idx]
        roi_mask = roi_masks[middle_slice_idx]
        
        # Normalize for display
        roi_slice_norm = (roi_slice - np.min(roi_slice)) / (np.max(roi_slice) - np.min(roi_slice) + 1e-8)
        
        # Plot 1: Original ROI
        axes[0].imshow(roi_slice_norm, cmap='gray')
        axes[0].set_title(f'Original ROI (Slice {middle_slice_idx})')
        axes[0].axis('off')
        
        # Plot 2: Mask overlay
        axes[1].imshow(roi_slice_norm, cmap='gray')
        mask_overlay = np.ma.masked_where(roi_mask == 0, roi_mask)
        axes[1].imshow(mask_overlay, cmap='jet', alpha=0.6)
        axes[1].set_title('ROI with Mask Overlay')
        axes[1].axis('off')
        
        # Plot 3: ROI bounds visualization
        x_min, y_min, x_max, y_max = roi_bounds
        axes[2].imshow(roi_slice_norm, cmap='gray')
        rect = Rectangle((0, 0), x_max-x_min, y_max-y_min, linewidth=2, edgecolor='red', facecolor='none')
        axes[2].add_patch(rect)
        axes[2].set_title(f'ROI Bounds: {x_max-x_min}x{y_max-y_min}')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None

def cleanup_temp_folder(temp_dir):
    """Clean up temporary directory"""
    try:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Error cleaning up temporary directory: {e}")

def find_sam2_configs():
    """Find available SAM2 config files"""
    configs = []
    try:
        import sam2
        sam2_path = os.path.dirname(sam2.__file__)
        
        # Common config locations
        config_locations = [
            os.path.join(sam2_path, "configs"),
            os.path.join(sam2_path, "configs", "sam2"),
            sam2_path
        ]
        
        for location in config_locations:
            if os.path.exists(location):
                for root, dirs, files in os.walk(location):
                    for file in files:
                        if file.endswith('.yaml') and 'hiera' in file.lower():
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, sam2_path)
                            configs.append((file, rel_path, full_path))
        
        return configs
    except Exception as e:
        print(f"Error finding configs: {e}")
        return []

class SAM2VideoManager:
    """Manager class for SAM2 video operations with propagation"""
    
    def __init__(self):
        self.video_predictor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        self.inference_state = None
    
    def load_video_model(self):
        """Load the SAM2 video predictor model"""
        if self.model_loaded:
            return True, "Video model already loaded"
        
        if not SAM2_VIDEO_AVAILABLE:
            return False, "SAM2 video predictor not available. Install SAM2 properly."
        
        try:
            # Checkpoint path
            checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
            
            if not os.path.exists(checkpoint_path):
                return False, f"Checkpoint not found: {checkpoint_path}"
            
            # Find available configs
            available_configs = find_sam2_configs()
            print(f"Found {len(available_configs)} SAM2 config files:")
            for name, rel_path, full_path in available_configs:
                print(f"  {name} -> {rel_path}")
            
            video_predictor = None
            config_used = None
            last_error = None
            
            # Try configs in order of preference
            preferred_configs = [
                "sam2_hiera_l.yaml",
                "sam2_hiera_large.yaml", 
                "sam2.1_hiera_l.yaml",
                "sam2.1_hiera_large.yaml"
            ]
            
            # First try with standard config names
            for config_name in preferred_configs:
                try:
                    video_predictor = build_sam2_video_predictor(config_name, checkpoint_path, device=self.device)
                    config_used = config_name
                    break
                except Exception as config_error:
                    last_error = str(config_error)
                    continue
            
            # If standard names fail, try found config files
            if video_predictor is None and available_configs:
                for name, rel_path, full_path in available_configs:
                    try:
                        # Try both relative path and full path
                        for config_path in [rel_path, full_path, name]:
                            try:
                                video_predictor = build_sam2_video_predictor(config_path, checkpoint_path, device=self.device)
                                config_used = f"{name} ({config_path})"
                                break
                            except:
                                continue
                        if video_predictor is not None:
                            break
                    except Exception as config_error:
                        last_error = str(config_error)
                        continue
            
            if video_predictor is None:
                error_msg = f"Failed to load video predictor with all configs.\n"
                error_msg += f"Last error: {last_error}\n\n"
                error_msg += f"Troubleshooting:\n"
                error_msg += f"1. Check SAM2 installation: pip list | grep sam2\n"
                error_msg += f"2. Check checkpoint exists: {checkpoint_path}\n"
                error_msg += f"3. Try reinstalling: pip install git+https://github.com/facebookresearch/segment-anything-2.git"
                return False, error_msg
            
            self.video_predictor = video_predictor
            self.model_loaded = True
            return True, f"SAM2 video predictor loaded successfully on device: {self.device} using config: {config_used}"
            
        except Exception as e:
            return False, f"Unexpected error loading video model: {str(e)}"
    
    def init_inference_state(self, video_dir_path):
        """Initialize inference state for video prediction using JPEG folder"""
        if not self.model_loaded:
            return False, "Video model not loaded"
        
        try:
            # SAM2 expects a directory path with JPEG files, not frames in memory
            self.inference_state = self.video_predictor.init_state(video_path=video_dir_path)
            return True, "Inference state initialized successfully"
        except Exception as e:
            return False, f"Error initializing inference state: {str(e)}"
    
    def add_new_box_and_get_mask(self, frame_idx, obj_id, box):
        """Add box and get mask predictions for SAM2 video"""
        if self.inference_state is None:
            return None, None, "Inference state not initialized"
        
        try:
            _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box,
            )
            return out_obj_ids, out_mask_logits, "Box added successfully"
        except Exception as e:
            return None, None, f"Error adding box: {str(e)}"
    
    def propagate_masks(self):
        """Propagate masks through video frames"""
        if self.inference_state is None:
            return None, "Inference state not initialized"
        
        try:
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(self.inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            return video_segments, "Propagation completed successfully"
        except Exception as e:
            return None, f"Error during propagation: {str(e)}"

def safe_normalize_image(img):
    """
    Safely normalize image to 0-255 range, handling edge cases
    """
    if img.size == 0:
        return np.array([], dtype=np.uint8)
    
    img_min = np.min(img)
    img_max = np.max(img)
    
    if img_max == img_min:
        # Handle case where all values are the same
        return np.zeros_like(img, dtype=np.uint8)
    
    # Normalize to 0-1 range then scale to 0-255
    img_norm = (img - img_min) / (img_max - img_min + 1e-8)
    return (img_norm * 255).astype(np.uint8)

def process_nifti_with_sam2_propagation(nifti_path, mask_data, threshold_data, output_dir, 
                                        status_placeholder=None, visualization_placeholder=None, progress_placeholder=None):
    """
    Process NIfTI file using SAM2 video propagation with real-time visualization
    
    Args:
        nifti_path: Path to the NIfTI file
        mask_data: Mask data from draw step (3D numpy array)
        threshold_data: Threshold parameters
        output_dir: Output directory for results
        status_placeholder: Streamlit placeholder for status updates
        visualization_placeholder: Streamlit placeholder for visualizations
        progress_placeholder: Streamlit placeholder for progress bars
    
    Returns:
        success: Boolean indicating success
        message: Status message
        results: Dictionary with processing results
    """
    
    def update_status(message, level="info"):
        """Helper function to update status"""
        if status_placeholder:
            if level == "info":
                status_placeholder.info(f"📋 {message}")
            elif level == "success":
                status_placeholder.success(f"✅ {message}")
            elif level == "warning":
                status_placeholder.warning(f"⚠️ {message}")
            elif level == "error":
                status_placeholder.error(f"❌ {message}")
        print(message)
    
    def update_progress(step, total_steps, current_step_name):
        """Helper function to update progress"""
        if progress_placeholder:
            # Ensure progress doesn't exceed 1.0
            progress = min(1.0, max(0.0, step / total_steps))
            progress_placeholder.progress(progress, text=f"Step {step}/{total_steps}: {current_step_name}")
    
    try:
        total_steps = 9  # Updated total processing steps
        current_step = 0
        
        current_step += 1
        update_status(f"Loading NIfTI file: {os.path.basename(nifti_path)}")
        update_progress(current_step, total_steps, "Loading NIfTI file")
        
        # Load NIfTI file
        nii_img = nib.load(nifti_path)
        nii_data = nii_img.get_fdata()
        
        if len(nii_data.shape) != 3:
            update_status("NIfTI file must be 3D", "error")
            return False, "NIfTI file must be 3D", None
        
        # Validate mask data
        if mask_data is None or mask_data.size == 0:
            update_status("Mask data is empty or None", "error")
            return False, "Mask data is empty or None", None
        
        current_step += 1
        update_status("Analyzing image and mask dimensions")
        update_progress(current_step, total_steps, "Analyzing dimensions")
        
        # Debug info
        print(f"NIfTI shape: {nii_data.shape}")
        print(f"Mask shape: {mask_data.shape}")
        print(f"Mask data type: {mask_data.dtype}")
        print(f"Mask non-zero count: {np.count_nonzero(mask_data) if mask_data.size > 0 else 0}")
        
        # Correct dimensions: NIfTI format is (slices, height, width) - first dimension is slices
        num_slices = nii_data.shape[0]  # Number of slices is the first dimension
        height = nii_data.shape[1]      # Height is second dimension  
        width = nii_data.shape[2]       # Width is third dimension
        
        print(f"Corrected interpretation: {num_slices} slices of {height}x{width} pixels")
        
        current_step += 1
        update_status(f"Extracting ROI from {num_slices} slices using traditional approach")
        update_progress(current_step, total_steps, "Extracting ROI")
        
        # Extract representative slice for ROI calculation (middle slice like traditional mode)
        middle_slice_idx = num_slices // 2  # Use middle slice like process_step
        middle_image_slice = nii_data[middle_slice_idx, :, :]  # Shape: (height, width)
        middle_mask_slice = mask_data[middle_slice_idx, :, :] if mask_data.shape[0] > middle_slice_idx else mask_data[0, :, :]
        
        print(f"Using middle slice {middle_slice_idx} for threshold application (like process_step)")
        
        # Apply threshold to middle slice EXACTLY like process_step does
        current_step += 1
        update_status(f"Applying threshold to middle slice {middle_slice_idx} (like process_step)")
        update_progress(current_step, total_steps, "Applying threshold")
        
        # Get threshold value (convert from dict format if needed)
        if isinstance(threshold_data, (int, float)):
            threshold_value = float(threshold_data)
        else:
            threshold_value = threshold_data.get('threshold', 0.5)
        
        print(f"Using threshold value: {threshold_value}")
        
        # Apply threshold EXACTLY like ThresholdOperations.threshold_image in process_step
        mn, mx = middle_image_slice.min(), middle_image_slice.max()
        if mx > mn:
            norm_middle = (middle_image_slice - mn) / (mx - mn)
        else:
            norm_middle = np.zeros_like(middle_image_slice)
        
        # Apply threshold: (normalized > threshold) & (mask > 0) - EXACT same logic
        thresholded_middle = (norm_middle > threshold_value) & (middle_mask_slice > 0)
        
        print(f"Threshold applied - pixels: {np.sum(thresholded_middle)} out of {thresholded_middle.size}")
        
        # Show threshold visualization before SAM2 processing
        if visualization_placeholder:
            try:
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # Original middle slice 
                axes[0].imshow(middle_image_slice, cmap='gray')
                axes[0].set_title(f'Original Middle Slice {middle_slice_idx}')
                axes[0].axis('off')
                
                # Normalized middle slice
                axes[1].imshow(norm_middle, cmap='gray') 
                axes[1].set_title(f'Normalized (like process_step)')
                axes[1].axis('off')
                
                # Mask
                axes[2].imshow(middle_mask_slice, cmap='gray')
                axes[2].set_title('Mask')
                axes[2].axis('off')
                
                # Thresholded result (what will go to SAM2)
                axes[3].imshow(thresholded_middle, cmap='gray')
                axes[3].set_title(f'Thresholded (T={threshold_value:.3f})')
                axes[3].axis('off')
                
                plt.tight_layout()
                
                with visualization_placeholder.container():
                    st.write("**Threshold Application (Process Step Logic):**")
                    st.pyplot(fig)
                    st.write(f"**Threshold Value:** {threshold_value:.3f} | **Pixels:** {np.sum(thresholded_middle):,} | **Middle Slice:** {middle_slice_idx}")
                
                plt.close()
                
            except Exception as viz_error:
                print(f"Threshold visualization error: {viz_error}")
        # Use traditional ROI extraction approach (but adapted for SAM2)
        roi_bounds, roi_slices, roi_masks, success = extract_roi_traditional_approach(
            nii_data, mask_data, middle_image_slice, middle_mask_slice, num_slices
        )
        if not success:
            update_status("Traditional ROI extraction failed", "error")
            return False, "Failed to extract valid region of interest using traditional approach", None
        
        current_step += 1
        update_status(f"ROI extracted successfully: {len(roi_slices)} slices")
        update_progress(current_step, total_steps, "Processing slices for SAM2")
        
        # Show ROI visualization
        if visualization_placeholder:
            try:
                roi_viz = create_sam2_progress_visualization(roi_slices, roi_masks, roi_bounds, middle_slice_idx)
                if roi_viz:
                    visualization_placeholder.pyplot(roi_viz)
            except Exception as viz_error:
                print(f"Visualization error: {viz_error}")
        
        print(f"ROI bounds: {roi_bounds}")
        print(f"Number of ROI slices: {len(roi_slices)}")
        
        # Process slices for SAM2
        processed_slices = []
        
        # Show threshold visualization for first slice
        if visualization_placeholder and len(roi_slices) > 0 and len(roi_masks) > 0:
            try:
                import matplotlib.pyplot as plt
                
                first_slice = roi_slices[0]
                first_mask = roi_masks[0]
                threshold_value = threshold_data.get('threshold', 0.5)  # Changed from 'upper_threshold' to 'threshold'
                
                # Create threshold visualization
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # Original ROI slice
                roi_norm = (first_slice - np.min(first_slice)) / (np.max(first_slice) - np.min(first_slice) + 1e-8)
                axes[0].imshow(roi_norm, cmap='gray')
                axes[0].set_title('Original ROI Slice')
                axes[0].axis('off')
                
                # Mask overlay
                axes[1].imshow(roi_norm, cmap='gray')
                mask_overlay = np.ma.masked_where(first_mask == 0, first_mask)
                axes[1].imshow(mask_overlay, cmap='jet', alpha=0.6)
                axes[1].set_title('ROI with Mask')
                axes[1].axis('off')
                
                # Apply threshold visualization
                if isinstance(threshold_value, (int, float)):
                    thresholded_preview = apply_safe_threshold(first_slice, first_mask, threshold_value)
                    axes[2].imshow(thresholded_preview, cmap='gray')
                    axes[2].set_title(f'Thresholded (T={threshold_value:.2f})')
                else:
                    normalized_preview = safe_normalize_for_sam2(first_slice)
                    axes[2].imshow(normalized_preview, cmap='gray')
                    axes[2].set_title('Normalized')
                axes[2].axis('off')
                
                # Create preview of what will be processed
                if isinstance(threshold_value, (int, float)):
                    preview_processed = apply_safe_threshold(first_slice, first_mask, threshold_value)
                else:
                    preview_processed = safe_normalize_for_sam2(first_slice)
                
                axes[3].imshow(preview_processed, cmap='gray')
                axes[3].set_title('Will be Processed')
                axes[3].axis('off')
                
                plt.tight_layout()
                
                # Update visualization before processing
                with visualization_placeholder.container():
                    st.pyplot(fig)
                    st.write(f"**Threshold Settings:** Upper = {threshold_value}")
                
            except Exception as viz_error:
                print(f"Threshold visualization error: {viz_error}")
        
        for slice_idx, (roi_slice, roi_mask) in enumerate(zip(roi_slices, roi_masks)):
            if slice_idx == middle_slice_idx:
                # Use the THRESHOLDED middle slice for the central frame (CRITICAL for SAM2)
                # Extract ROI from the thresholded middle slice
                x_min, y_min, x_max, y_max = roi_bounds
                thresholded_middle_roi = thresholded_middle[y_min:y_max, x_min:x_max]
                processed_slices.append(thresholded_middle_roi.astype(np.uint8))
                print(f"✅ Added THRESHOLDED middle slice {slice_idx} as central frame for SAM2")
            else:
                # For other slices, just normalize them (they will be propagated by SAM2)
                normalized_slice = safe_normalize_for_sam2(roi_slice)
                processed_slices.append(normalized_slice)
                print(f"Added normalized slice {slice_idx}")
        
        print(f"Total processed slices: {len(processed_slices)}, Central thresholded slice: {middle_slice_idx}")
        
        # Convert to format suitable for SAM2 video predictor using JPEG folder approach
        current_step += 1
        update_status("Creating temporary JPEG folder for SAM2...")
        update_progress(current_step, total_steps, "Creating JPEG files")
        
        print("Creating temporary JPEG folder for SAM2...")
        temp_video_dir, frame_names, jpeg_success = create_temp_jpeg_folder(processed_slices)
        
        if not jpeg_success:
            update_status("Failed to create temporary JPEG folder", "error")
            return False, "Failed to create temporary JPEG folder for SAM2", None
        
        update_status(f"Created {len(frame_names)} JPEG frames successfully")
        
        # Show sample frame visualization
        if visualization_placeholder and len(processed_slices) > 0:
            try:
                import matplotlib.pyplot as plt
                
                # Show first few frames as samples
                num_samples = min(4, len(processed_slices))
                fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
                
                if num_samples == 1:
                    axes = [axes]
                
                for i in range(num_samples):
                    slice_data = processed_slices[i]
                    axes[i].imshow(slice_data, cmap='gray')
                    axes[i].set_title(f'Frame {i:05d}.jpg')
                    axes[i].axis('off')
                
                plt.suptitle(f'Sample JPEG Frames (Total: {len(frame_names)})')
                plt.tight_layout()
                
                with visualization_placeholder.container():
                    st.pyplot(fig)
                    st.write(f"**JPEG Conversion:** {len(frame_names)} frames created for SAM2 video processing")
                
            except Exception as viz_error:
                print(f"JPEG visualization error: {viz_error}")
        
        try:
            current_step += 1
            update_status("Loading SAM2 video model...")
            update_progress(current_step, total_steps, "Loading SAM2 model")
            
            # Initialize SAM2 video manager with enhanced error handling
            sam2_video = SAM2VideoManager()
            
            # Load video model with improved config handling
            print("Loading SAM2 video model...")
            success, message = sam2_video.load_video_model()
            if not success:
                print(f"SAM2 video model loading failed: {message}")
                # Try to provide more specific error information
                if "config" in message.lower():
                    print("Config-related error detected. Checking available configs...")
                    configs = find_sam2_configs()
                    if configs:
                        print(f"Found {len(configs)} potential configs:")
                        for name, rel_path, full_path in configs[:3]:
                            print(f"  - {name} at {rel_path}")
                    else:
                        print("No suitable config files found in SAM2 installation")
                
                cleanup_temp_folder(temp_video_dir)
                update_status(f"Failed to load SAM2 video model: {message}", "error")
                return False, f"Failed to load SAM2 video model: {message}", None
            
            update_status("SAM2 video model loaded successfully", "success")
            print(f"SAM2 video model loaded successfully: {message}")
            
            current_step += 1
            update_status("Initializing SAM2 inference state...")
            update_progress(current_step, total_steps, "Initializing inference")
            
            # Initialize inference state with JPEG folder path
            success, message = sam2_video.init_inference_state(temp_video_dir)
            if not success:
                cleanup_temp_folder(temp_video_dir)
                update_status(f"Failed to initialize inference state: {message}", "error")
                return False, f"Failed to initialize inference state: {message}", None
            
            update_status("Inference state initialized successfully", "success")
            print("Inference state initialized successfully")
            
            # Reset state for clean start
            sam2_video.video_predictor.reset_state(sam2_video.inference_state)
            
            # Create bounding box from the THRESHOLDED middle slice for the central frame
            # Use the ROI bounds we calculated from the middle slice
            x_min, y_min, x_max, y_max = roi_bounds
            thresholded_middle_roi = thresholded_middle[y_min:y_max, x_min:x_max]
            
            # Find bounding box coordinates from the thresholded middle ROI
            y_coords, x_coords = np.where(thresholded_middle_roi > 0)
            if len(x_coords) == 0 or len(y_coords) == 0:
                cleanup_temp_folder(temp_video_dir)
                return False, "No valid thresholded region found in middle slice ROI", None
            
            # Create bounding box with some padding
            padding = 5
            bbox_x_min = max(0, np.min(x_coords) - padding)
            bbox_y_min = max(0, np.min(y_coords) - padding)
            bbox_x_max = min(thresholded_middle_roi.shape[1], np.max(x_coords) + padding)
            bbox_y_max = min(thresholded_middle_roi.shape[0], np.max(y_coords) + padding)
            
            # SAM2 box format: [x_min, y_min, x_max, y_max]
            bbox = np.array([bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max], dtype=np.float32)
            
            print(f"Using bounding box from thresholded middle slice: {bbox}")
            
            # Add box to MIDDLE frame (not first frame) - this is the thresholded central slice
            frame_idx = middle_slice_idx  # Use middle slice as central frame
            obj_id = 1
            
            # Show SAM2 inference visualization BEFORE propagation
            if visualization_placeholder:
                try:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Show the thresholded middle slice that will receive the box prompt
                    axes[0].imshow(thresholded_middle_roi, cmap='gray')
                    rect = plt.Rectangle((bbox_x_min, bbox_y_min), bbox_x_max-bbox_x_min, bbox_y_max-bbox_y_min, 
                                       fill=False, color='red', linewidth=2)
                    axes[0].add_patch(rect)
                    axes[0].set_title(f'Central Frame {middle_slice_idx} + Box Prompt')
                    axes[0].axis('off')
                    
                    # Show just the bounding box region
                    bbox_region = thresholded_middle_roi[bbox_y_min:bbox_y_max, bbox_x_min:bbox_x_max]
                    axes[1].imshow(bbox_region, cmap='gray')
                    axes[1].set_title('Box Region (SAM2 Input)')
                    axes[1].axis('off')
                    
                    # Placeholder for SAM2 result (will be updated after inference)
                    axes[2].text(0.5, 0.5, 'SAM2 Result\n(Processing...)', 
                               ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
                    axes[2].set_title('SAM2 Inference Result')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    
                    with visualization_placeholder.container():
                        st.write("**🤖 SAM2 Box Prompt Inference:**")
                        st.pyplot(fig)
                        st.write(f"**Central Frame:** {middle_slice_idx} | **Box:** {bbox} | **Threshold:** {threshold_value:.3f}")
                    
                    plt.close()
                    
                except Exception as viz_error:
                    print(f"SAM2 inference visualization error: {viz_error}")
            
            out_obj_ids, out_mask_logits, box_message = sam2_video.add_new_box_and_get_mask(
                frame_idx, obj_id, bbox
            )
            
            if out_obj_ids is None:
                cleanup_temp_folder(temp_video_dir)
                return False, f"Failed to add bounding box: {box_message}", None
            
            print(f"Bounding box added successfully to frame {frame_idx}: {box_message}")
            
            # Show SAM2 inference result
            if visualization_placeholder and out_mask_logits is not None:
                try:
                    # Get the SAM2 result mask for the central frame
                    sam2_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                    
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    
                    # Original thresholded 
                    axes[0].imshow(thresholded_middle_roi, cmap='gray')
                    axes[0].set_title(f'Original Thresholded {middle_slice_idx}')
                    axes[0].axis('off')
                    
                    # Box prompt visualization
                    axes[1].imshow(thresholded_middle_roi, cmap='gray')
                    rect = plt.Rectangle((bbox_x_min, bbox_y_min), bbox_x_max-bbox_x_min, bbox_y_max-bbox_y_min, 
                                       fill=False, color='red', linewidth=2)
                    axes[1].add_patch(rect)
                    axes[1].set_title('Box Prompt')
                    axes[1].axis('off')
                    
                    # SAM2 result
                    axes[2].imshow(sam2_mask, cmap='gray')
                    axes[2].set_title('SAM2 Inference Result')
                    axes[2].axis('off')
                    
                    # Overlay comparison
                    axes[3].imshow(thresholded_middle_roi, cmap='gray', alpha=0.7)
                    axes[3].imshow(sam2_mask, cmap='jet', alpha=0.5)
                    axes[3].set_title('Overlay: Gray=Original, Color=SAM2')
                    axes[3].axis('off')
                    
                    plt.tight_layout()
                    
                    with visualization_placeholder.container():
                        st.write("**✅ SAM2 Inference Complete:**")
                        st.pyplot(fig)
                        st.write(f"**SAM2 pixels:** {np.sum(sam2_mask):,} | **Original pixels:** {np.sum(thresholded_middle_roi):,}")
                    
                    plt.close()
                    
                except Exception as viz_error:
                    print(f"SAM2 result visualization error: {viz_error}")
            
            # Propagate masks through all frames
            current_step += 1
            update_status("Starting SAM2 mask propagation...")
            update_progress(current_step, total_steps, "Propagating masks")
            
            print("Starting mask propagation...")
            video_segments, prop_message = sam2_video.propagate_masks()
            if video_segments is None:
                cleanup_temp_folder(temp_video_dir)
                update_status(f"Failed to propagate masks: {prop_message}", "error")
                return False, f"Failed to propagate masks: {prop_message}", None
            
            update_status(f"Mask propagation completed successfully", "success")
            print(f"Mask propagation completed: {prop_message}")
            print(f"Processed frames: {sorted(video_segments.keys())}")
            
            # Show propagation results visualization
            if visualization_placeholder and len(video_segments) > 0:
                try:
                    import matplotlib.pyplot as plt
                    
                    # Show sample propagated masks
                    sample_frames = sorted(list(video_segments.keys()))[:4]  # First 4 frames
                    if len(sample_frames) > 0:
                        fig, axes = plt.subplots(1, len(sample_frames), figsize=(15, 4))
                        
                        if len(sample_frames) == 1:
                            axes = [axes]
                        
                        for i, frame_idx in enumerate(sample_frames):
                            if 1 in video_segments[frame_idx]:  # Object ID 1
                                mask = video_segments[frame_idx][1]
                                axes[i].imshow(mask, cmap='jet')
                                axes[i].set_title(f'Frame {frame_idx}')
                                axes[i].axis('off')
                        
                        plt.suptitle(f'SAM2 Propagated Masks (Total: {len(video_segments)} frames)')
                        plt.tight_layout()
                        
                        with visualization_placeholder.container():
                            st.pyplot(fig)
                            st.write(f"**SAM2 Results:** Successfully propagated to {len(video_segments)} frames")
                
                except Exception as viz_error:
                    print(f"Results visualization error: {viz_error}")
            
            # Convert results back to full image space (traditional approach)
            current_step += 1
            update_status("Converting results back to full image space...")
            update_progress(current_step, total_steps, "Converting results")
            
            output_masks = []
            x_min, y_min, x_max, y_max = roi_bounds
            
            # Process each slice like traditional mode
            for slice_idx in range(nii_data.shape[0]):  # First dimension is number of slices
                # Create full-size mask for this slice
                full_mask = np.zeros((nii_data.shape[1], nii_data.shape[2]), dtype=np.uint8)  # (height, width)
                
                # Check if we have a propagated mask for this slice
                if slice_idx in video_segments and 1 in video_segments[slice_idx]:
                    prop_mask = video_segments[slice_idx][1]  # Object ID 1
                    
                    # Debug: Check propagated mask dimensions
                    print(f"Slice {slice_idx}: propagated mask shape: {prop_mask.shape}")
                    
                    # Ensure propagated mask is the right size for ROI
                    roi_h = y_max - y_min
                    roi_w = x_max - x_min
                    
                    print(f"ROI dimensions: {roi_w}x{roi_h}, mask dimensions: {prop_mask.shape}")
                    
                    # Validate dimensions before resize
                    if roi_h <= 0 or roi_w <= 0:
                        print(f"Invalid ROI dimensions for slice {slice_idx}: {roi_w}x{roi_h}")
                        continue
                    
                    if prop_mask.shape[0] <= 0 or prop_mask.shape[1] <= 0:
                        print(f"Invalid mask dimensions for slice {slice_idx}: {prop_mask.shape}")
                        continue
                    
                    # Only resize if dimensions don't match
                    if prop_mask.shape[0] != roi_h or prop_mask.shape[1] != roi_w:
                        try:
                            prop_mask = cv2.resize(prop_mask.astype(np.uint8), (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                            print(f"Resized mask from {prop_mask.shape} to {roi_h}x{roi_w}")
                        except Exception as resize_error:
                            print(f"Resize failed for slice {slice_idx}: {resize_error}")
                            continue
                    
                    # Map back to full image
                    full_mask[y_min:y_max, x_min:x_max] = (prop_mask > 0.5).astype(np.uint8)
                    print(f"Mapped mask to full image for slice {slice_idx}")
                
                output_masks.append(full_mask)
            
        finally:
            # Clean up temporary directory
            cleanup_temp_folder(temp_video_dir)
        
        # Save results EXACTLY like batch_process_step
        current_step += 1
        update_status("Saving results in same format as process_step...")
        update_progress(current_step, total_steps, "Saving results")
        
        # Create save directory structure EXACTLY like batch_process_step  
        filename_no_ext = os.path.splitext(os.path.basename(nifti_path))[0]
        output_path = os.path.join(output_dir, filename_no_ext)
        save_dir = os.path.join(output_path, 'sam_dense_mask')  # Use sam_dense_mask to differentiate from process_step
        
        # Clear existing output directory (like batch_process_step)
        if os.path.exists(save_dir):
            for f in os.listdir(save_dir):
                os.remove(os.path.join(save_dir, f))
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Saving results to: {save_dir}")
        
        # Save individual PNG files EXACTLY like batch_process_step
        for slice_idx, full_mask in enumerate(output_masks):
            # Convert to binary image like batch_process_step: np.where(thresholded_image > 0, 255, 0)
            binary_image = np.where(full_mask > 0, 255, 0).astype(np.uint8)
            
            # Use same filename format as batch_process_step: f'slice_{slice_index}_threshold_{threshold:.4f}.png'
            filename = f'slice_{slice_idx}_sam2_threshold_{threshold_value:.4f}.png'
            filepath = os.path.join(save_dir, filename)
            
            # Save with transpose like batch_process_step: Image.fromarray(binary_image.T, mode='L')
            Image.fromarray(binary_image.T, mode='L').save(filepath)
            
            print(f"Saved slice {slice_idx}: {filename}")
        
        # Create NIfTI file EXACTLY like batch_process_step using MaskOperations.create_mask_nifti
        try:
            nifti_path_result = MaskOperations.create_mask_nifti(save_dir, nii_img.affine)
            print(f"✅ Created NIfTI file: {nifti_path_result}")
        except Exception as nifti_error:
            print(f"Error creating NIfTI file: {nifti_error}")
        
        # Prepare results with same structure
        results = {
            "save_dir": save_dir,
            "roi_bounds": roi_bounds,
            "num_slices": len(output_masks),
            "propagation_frames": len(video_segments),
            "nifti_file": nifti_path_result if 'nifti_path_result' in locals() else None
        }
        
        update_status(f"SAM2 processing completed successfully!", "success")
        update_progress(total_steps, total_steps, "Complete")
        
        print(f"✅ SAM2 processing completed successfully")
        print(f"Results saved to: {save_dir}")
        print(f"Format: {len(output_masks)} PNG files + mask.nii (same as process_step)")
        
        return True, "SAM2 processing completed successfully", results
        
    except Exception as e:
        return False, f"Error processing with SAM2: {str(e)}", None

def create_sam2_visualization(original_slices, video_segments, mask_bounds, output_path):
    """Create visualization of SAM2 propagation results"""
    try:
        num_slices = len(original_slices)
        cols = min(8, num_slices)
        rows = (num_slices + cols - 1) // cols
        
        # Calculate dimensions for the combined image
        slice_size = 150  # Size for each slice in pixels
        fig_width = cols * slice_size
        fig_height = rows * slice_size
        
        # Create combined image
        combined_img = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)
        
        for i, slice_data in enumerate(original_slices):
            row = i // cols
            col = i % cols
            
            # Normalize slice
            slice_norm = safe_normalize_image(slice_data)
            
            # Resize slice to fit the grid
            slice_resized = cv2.resize(slice_norm, (slice_size, slice_size))
            
            # Get corresponding mask if available
            mask = None
            if i in video_segments and 1 in video_segments[i]:
                mask = video_segments[i][1]
                # Resize mask to match slice
                mask_resized = cv2.resize(mask.astype(np.uint8), (slice_size, slice_size))
            
            # Create overlay
            if mask is not None:
                # Create colored overlay
                overlay = np.stack([slice_resized] * 3, axis=-1)
                
                # Add mask in red with transparency
                mask_indices = mask_resized > 0.5
                overlay[mask_indices] = [255, slice_resized[mask_indices] // 2, slice_resized[mask_indices] // 2]
            else:
                overlay = np.stack([slice_resized] * 3, axis=-1)
            
            # Place in combined image
            y_start = row * slice_size
            y_end = y_start + slice_size
            x_start = col * slice_size
            x_end = x_start + slice_size
            
            combined_img[y_start:y_end, x_start:x_end] = overlay
        
        # Save combined visualization
        Image.fromarray(combined_img).save(output_path)
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")

def batch_sam2_process_step():
    """SAM2 batch processing step"""
    st.header("🤖 Step 4: SAM2 Processing")
    
    # Add consistent CSS styling
    st.markdown("""
    <style>
    .step-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2d3436;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(168, 237, 234, 0.3);
        border-left: 3px solid #00b894;
    }
    .sam2-info {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(116, 185, 255, 0.3);
    }
    .progress-section {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        box-shadow: 0 1px 5px rgba(0,0,0,0.05);
    }
    .file-status {
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    .status-success {
        background: linear-gradient(135deg, #00b894 0%, #55efc4 100%);
        color: white;
    }
    .status-error {
        background: linear-gradient(135deg, #e17055 0%, #fdcb6e 100%);
        color: white;
    }
    .status-processing {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # SAM2 Mode Information
    st.markdown("""
    <div class="sam2-info">
        <h3>🤖 SAM2 Video Propagation Mode</h3>
        <p>This mode uses SAM2's video predictor for advanced segmentation propagation:</p>
        <ul>
            <li><strong>First Frame:</strong> Uses threshold + SAM2 inference for initial segmentation</li>
            <li><strong>Propagation:</strong> SAM2 automatically segments remaining slices based on the first frame</li>
            <li><strong>Consistency:</strong> Maintains temporal consistency across all slices</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if required data is available
    if "batch_files" not in st.session_state or "batch_final_thresholds" not in st.session_state:
        st.error("Missing required data. Please complete the previous steps.")
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        return
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check SAM2 availability and run diagnostics
    st.markdown('<div class="progress-section">', unsafe_allow_html=True)
    st.markdown("### 🔧 SAM2 System Check")
    
    with st.expander("View SAM2 Diagnostic Info", expanded=False):
        if st.button("Run SAM2 Diagnostic"):
            with st.spinner("Running SAM2 diagnostic..."):
                diagnostic_success = test_sam2_installation()
                if diagnostic_success:
                    st.success("✅ SAM2 installation check passed!")
                else:
                    st.error("❌ SAM2 installation issues detected. Check console output.")
    
    sam2_manager = SAM2Manager()
    deps_ok, deps_msg = sam2_manager.check_dependencies()
    checkpoint_ok, checkpoint_msg = sam2_manager.check_checkpoint()
    
    st.markdown(f"**Dependencies:** {'✅' if deps_ok else '❌'} {deps_msg}")
    st.markdown(f"**Checkpoint:** {'✅' if checkpoint_ok else '❌'} {checkpoint_msg}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not deps_ok or not checkpoint_ok:
        st.error("SAM2 Setup Required")
        st.markdown(f"**Dependencies:** {deps_msg}")
        st.markdown(f"**Checkpoint:** {checkpoint_msg}")
        
        if not checkpoint_ok:
            st.info("""
            **Download SAM2 Checkpoint:**
            ```bash
            mkdir -p checkpoints
            wget -O checkpoints/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
            ```
            """)
        
        if st.button("← Back to Threshold Step"):
            st.session_state["current_step"] = "batch_threshold"
            st.rerun()
        return
    
    # Initialize processing state
    if "sam2_processing_started" not in st.session_state:
        st.session_state["sam2_processing_started"] = False
        st.session_state["sam2_completed_files"] = {}
        st.session_state["sam2_current_file_idx"] = 0
    
    files = st.session_state["batch_files"]
    thresholds = st.session_state["batch_final_thresholds"]
    
    # Progress overview
    st.markdown('<div class="progress-section">', unsafe_allow_html=True)
    st.markdown("### 📊 Processing Progress")
    
    total_files = len(files)
    completed_files = len(st.session_state["sam2_completed_files"])
    progress_percentage = (completed_files / total_files) * 100
    
    st.progress(progress_percentage / 100)
    st.markdown(f"**Progress:** {completed_files}/{total_files} files completed ({progress_percentage:.1f}%)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if not st.session_state["sam2_processing_started"]:
            if st.button("🚀 Start SAM2 Processing", type="primary", use_container_width=True):
                st.session_state["sam2_processing_started"] = True
                st.rerun()
    
    with col2:
        if completed_files > 0:
            if st.button("💾 Save Results", use_container_width=True):
                save_sam2_results()
                st.success("Results saved successfully!")
    
    with col3:
        if st.button("← Back to Threshold", use_container_width=True):
            st.session_state["current_step"] = "batch_threshold"
            st.rerun()
    
    # Process files
    if st.session_state["sam2_processing_started"]:
        current_idx = st.session_state["sam2_current_file_idx"]
        
        if current_idx < total_files:
            # Get file info - batch_files contains strings (filenames)
            filename = files[current_idx]
            filename_no_ext = filename.split('.')[0]
            
            # Build file path
            input_folder = os.path.join(os.getcwd(), 'media')
            file_path = os.path.join(input_folder, filename)
            
            st.markdown(f'<div class="file-status status-processing">🤖 Processing: {filename}</div>', unsafe_allow_html=True)
            
            # Get mask data from saved files (created in draw step)
            output_path = os.path.join(os.getcwd(), 'output', filename_no_ext)
            mask_path = os.path.join(output_path, 'dense.nii')
            
            if not os.path.exists(mask_path):
                st.error(f"Mask file not found: {mask_path}")
                st.session_state["sam2_completed_files"][filename] = {
                    "status": "error",
                    "message": "Mask file not found - draw step may not be completed"
                }
            else:
                # Load mask data from file
                try:
                    import nibabel as nib
                    mask_nii = nib.load(mask_path)
                    mask_data = mask_nii.get_fdata()
                    
                    # Debug mask data with visualizations
                    st.write(f"**📊 Mask Analysis for {filename}:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Shape", f"{mask_data.shape}")
                    with col2:
                        st.metric("Data Type", f"{mask_data.dtype}")
                    with col3:
                        st.metric("Non-zero pixels", f"{np.count_nonzero(mask_data):,}")
                    
                    # Show mask value range
                    mask_min, mask_max = np.min(mask_data), np.max(mask_data)
                    st.write(f"Value range: [{mask_min:.3f}, {mask_max:.3f}]")
                    
                    # Validate mask data
                    if mask_data.size == 0:
                        st.error(f"Loaded mask is empty for {filename}")
                        st.session_state["sam2_completed_files"][filename] = {
                            "status": "error",
                            "message": "Loaded mask is empty"
                        }
                    elif np.count_nonzero(mask_data) == 0:
                        st.error(f"Loaded mask contains no non-zero values for {filename}")
                        st.session_state["sam2_completed_files"][filename] = {
                            "status": "error",
                            "message": "Mask contains no non-zero values"
                        }
                    else:
                        # Get threshold data from session state
                        threshold_data = thresholds.get(filename_no_ext, {})
                        
                        if not threshold_data:
                            st.error(f"No threshold data found for {filename}")
                            st.session_state["sam2_completed_files"][filename] = {
                                "status": "error", 
                                "message": "No threshold data found"
                            }
                        else:
                            # Convert threshold data to proper format if needed
                            if isinstance(threshold_data, (int, float)):
                                # threshold_data is the direct threshold value from batch_threshold_step
                                threshold_dict = {
                                    'threshold': float(threshold_data)  # Use 'threshold' key to match our corrected code
                                }
                            else:
                                # threshold_data is already a dict, ensure it has 'threshold' key
                                threshold_dict = threshold_data
                                if 'threshold' not in threshold_dict and 'upper_threshold' in threshold_dict:
                                    threshold_dict['threshold'] = threshold_dict['upper_threshold']
                            
                            # Create progress container for detailed visualization
                            progress_container = st.container()
                            
                            with progress_container:
                                st.write("### 🔍 SAM2 Processing Details")
                                
                                # Create placeholder for dynamic updates
                                status_placeholder = st.empty()
                                visualization_placeholder = st.empty()
                                progress_placeholder = st.empty()
                                
                                # Update status
                                with status_placeholder.container():
                                    st.info(f"🚀 Starting SAM2 processing for {filename}")
                            
                            # Process with SAM2 and pass visualization containers
                            success, message, results = process_nifti_with_sam2_propagation(
                                file_path, mask_data, threshold_dict, output_dir,
                                status_placeholder, visualization_placeholder, progress_placeholder
                            )
                            
                            st.session_state["sam2_completed_files"][filename] = {
                                "status": "success" if success else "error",
                                "message": message,
                                "results": results
                            }
                            
                            if success:
                                st.success(f"✅ {filename}: {message}")
                            else:
                                st.error(f"❌ {filename}: {message}")
                            
                except Exception as e:
                    st.error(f"Error loading mask data: {str(e)}")
                    st.session_state["sam2_completed_files"][filename] = {
                        "status": "error",
                        "message": f"Error loading mask data: {str(e)}"
                    }
            
            # Move to next file
            st.session_state["sam2_current_file_idx"] += 1
            st.rerun()
        
        else:
            # All files processed
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            st.markdown("### 🎉 SAM2 Processing Complete!")
            st.markdown(f"Successfully processed {total_files} files using SAM2 video propagation.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show completion statistics
            success_count = sum(1 for result in st.session_state["sam2_completed_files"].values() 
                              if result["status"] == "success")
            error_count = total_files - success_count
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("✅ Successful", success_count)
            with col2:
                st.metric("❌ Errors", error_count)
    
    # Show detailed results
    if st.session_state["sam2_completed_files"]:
        st.markdown('<div class="progress-section">', unsafe_allow_html=True)
        st.markdown("### 📋 Detailed Results")
        
        for filename, result in st.session_state["sam2_completed_files"].items():
            status_class = "status-success" if result["status"] == "success" else "status-error"
            status_icon = "✅" if result["status"] == "success" else "❌"
            
            st.markdown(f'<div class="file-status {status_class}">{status_icon} {filename}: {result["message"]}</div>', 
                       unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def save_sam2_results():
    """Save SAM2 processing results"""
    try:
        output_dir = "output"
        results_file = os.path.join(output_dir, "sam2_processing_results.json")
        
        # Prepare results for JSON serialization
        json_results = {}
        for filename, result in st.session_state["sam2_completed_files"].items():
            json_results[filename] = {
                "status": result["status"],
                "message": result["message"],
                "has_results": result.get("results") is not None
            }
        
        with open(results_file, 'w') as f:
            import json
            json.dump(json_results, f, indent=2)
        
        st.success(f"Results saved to {results_file}")
        
    except Exception as e:
        st.error(f"Error saving results: {str(e)}")
