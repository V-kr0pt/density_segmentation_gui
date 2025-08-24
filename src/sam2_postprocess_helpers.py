"""
SAM2 Post-Processing Helper Functions

This module contains helper functions for the SAM2 post-processing mode,
which uses existing batch_process_step PNG results as input for SAM2 video propagation.
"""

import os
import tempfile
import shutil
import numpy as np
from PIL import Image
import streamlit as st


def convert_png_to_jpeg_for_sam2(dense_mask_dir, png_files):
    """
    Convert PNG files from batch_process_step to JPEG format for SAM2.
    
    Args:
        dense_mask_dir: Directory containing PNG files
        png_files: List of PNG filenames
        
    Returns:
        success: Boolean indicating success
        temp_dir: Temporary directory path with JPEG files
        png_images: List of loaded PNG images as numpy arrays
    """
    try:
        # Create temporary directory for JPEG files
        temp_dir = tempfile.mkdtemp(prefix="sam2_postprocess_")
        png_images = []
        
        for i, png_file in enumerate(png_files):
            png_path = os.path.join(dense_mask_dir, png_file)
            
            # Load PNG image
            pil_image = Image.open(png_path)
            png_array = np.array(pil_image)
            
            # Ensure grayscale
            if len(png_array.shape) == 3:
                png_array = png_array[:, :, 0]  # Take first channel
            
            # Convert to RGB for SAM2 (SAM2 expects 3-channel images)
            png_rgb = np.stack([png_array] * 3, axis=-1)
            
            # Save as JPEG with zero-padded filename
            jpeg_filename = f"{i:05d}.jpg"
            jpeg_path = os.path.join(temp_dir, jpeg_filename)
            
            pil_rgb = Image.fromarray(png_rgb)
            pil_rgb.save(jpeg_path, "JPEG", quality=95)
            
            png_images.append(png_array)
        
        print(f"✅ Converted {len(png_files)} PNG files to JPEG format")
        print(f"Temporary JPEG directory: {temp_dir}")
        
        return True, temp_dir, png_images
        
    except Exception as e:
        print(f"Error converting PNG to JPEG: {e}")
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        return False, None, None


def initialize_sam2_video_predictor(temp_video_dir):
    """
    Initialize SAM2 video predictor with JPEG directory.
    
    Args:
        temp_video_dir: Directory containing JPEG files
        
    Returns:
        success: Boolean indicating success
        sam2_video: SAM2VideoManager instance
        message: Status message
    """
    try:
        from sam_utils import SAM2VideoManager, find_sam2_configs
        
        sam2_video = SAM2VideoManager()
        success, message = sam2_video.load_model()
        
        if not success:
            return False, None, f"Failed to load SAM2 model: {message}"
        
        # Initialize inference state
        success, message = sam2_video.init_inference_state(temp_video_dir)
        if not success:
            return False, None, f"Failed to initialize inference state: {message}"
        
        # Reset state for clean start
        sam2_video.video_predictor.reset_state(sam2_video.inference_state)
        
        return True, sam2_video, "SAM2 video predictor initialized successfully"
        
    except Exception as e:
        return False, None, f"Error initializing SAM2: {str(e)}"


def apply_box_prompt_to_first_slice(sam2_video, first_png_image):
    """
    Apply box prompt to the first slice for SAM2 inference.
    
    Args:
        sam2_video: SAM2VideoManager instance
        first_png_image: First PNG image as numpy array
        
    Returns:
        success: Boolean indicating success
        bbox: Bounding box coordinates
        message: Status message
    """
    try:
        # Find bounding box from first PNG image (already thresholded)
        y_coords, x_coords = np.where(first_png_image > 0)
        
        if len(x_coords) == 0 or len(y_coords) == 0:
            return False, None, "No valid region found in first slice"
        
        # Create bounding box with padding
        padding = 5
        bbox_x_min = max(0, np.min(x_coords) - padding)
        bbox_y_min = max(0, np.min(y_coords) - padding)
        bbox_x_max = min(first_png_image.shape[1], np.max(x_coords) + padding)
        bbox_y_max = min(first_png_image.shape[0], np.max(y_coords) + padding)
        
        # SAM2 box format: [x_min, y_min, x_max, y_max]
        bbox = np.array([bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max], dtype=np.float32)
        
        # Apply box prompt to frame 0
        frame_idx = 0
        obj_id = 1
        
        out_obj_ids, out_mask_logits, box_message = sam2_video.add_new_box_and_get_mask(
            frame_idx, obj_id, bbox
        )
        
        if out_obj_ids is None:
            return False, bbox, f"Failed to add box prompt: {box_message}"
        
        return True, bbox, f"Box prompt applied successfully: {box_message}"
        
    except Exception as e:
        return False, None, f"Error applying box prompt: {str(e)}"


def run_sam2_propagation(sam2_video):
    """
    Run SAM2 video propagation.
    
    Args:
        sam2_video: SAM2VideoManager instance
        
    Returns:
        success: Boolean indicating success
        video_segments: Propagation results
        message: Status message
    """
    try:
        video_segments, prop_message = sam2_video.propagate_masks()
        
        if video_segments is None:
            return False, None, f"Propagation failed: {prop_message}"
        
        return True, video_segments, f"Propagation successful: {prop_message}"
        
    except Exception as e:
        return False, None, f"Error in propagation: {str(e)}"


def visualize_box_on_image(image, bbox):
    """
    Create visualization of image with box overlay.
    
    Args:
        image: Input image as numpy array
        bbox: Bounding box coordinates [x_min, y_min, x_max, y_max]
        
    Returns:
        image_with_box: Image with box overlay
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from io import BytesIO
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(image, cmap='gray')
        
        # Add box
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                               linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        ax.set_title('Box Prompt Applied')
        ax.axis('off')
        
        # Convert to image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        
        # Load as PIL image
        result_image = Image.open(buf)
        plt.close()
        
        return result_image
        
    except Exception as e:
        print(f"Error creating box visualization: {e}")
        return Image.fromarray(image)


def save_sam2_results(nifti_path, output_dir, video_segments, png_images, temp_video_dir, visualization_placeholder):
    """
    Save SAM2 post-processing results in the same format as batch_process_step.
    
    Args:
        nifti_path: Original NIfTI file path
        output_dir: Output directory
        video_segments: SAM2 propagation results
        png_images: Original PNG images
        temp_video_dir: Temporary directory
        visualization_placeholder: Streamlit placeholder
        
    Returns:
        success: Boolean indicating success
        save_dir: Save directory path
        message: Status message
    """
    try:
        from utils import MaskOperations
        
        # Create save directory
        filename_no_ext = os.path.splitext(os.path.basename(nifti_path))[0]
        output_path = os.path.join(output_dir, filename_no_ext)
        save_dir = os.path.join(output_path, 'sam_mask')
        
        # Clear and create directory
        if os.path.exists(save_dir):
            for f in os.listdir(save_dir):
                os.remove(os.path.join(save_dir, f))
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert SAM2 results back to PNG format
        saved_count = 0
        for frame_idx in sorted(video_segments.keys()):
            if 1 in video_segments[frame_idx]:  # Object ID 1
                mask = video_segments[frame_idx][1]
                
                # Convert mask to binary image (0 or 255)
                binary_mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)
                
                # Use same filename format as batch_process_step
                filename = f'slice_{frame_idx}_sam2_postprocess.png'
                filepath = os.path.join(save_dir, filename)
                
                # Save with transpose (same as batch_process_step)
                Image.fromarray(binary_mask.T, mode='L').save(filepath)
                saved_count += 1
        
        # Create NIfTI file
        try:
            import nibabel
            nii_img = nibabel.load(nifti_path)
            nifti_path_result = MaskOperations.create_mask_nifti(save_dir, nii_img.affine)
        except Exception as nifti_error:
            print(f"Warning: Could not create NIfTI file: {nifti_error}")
            nifti_path_result = None
        
        # Show save results
        if visualization_placeholder:
            with visualization_placeholder.container():
                st.markdown("### 💾 **Results Saved**")
                st.success(f"Saved {saved_count} SAM2 processed slices")
                st.write(f"**Directory:** `{save_dir}`")
                if nifti_path_result:
                    st.write(f"**NIfTI file:** `{os.path.basename(nifti_path_result)}`")
        
        return True, save_dir, f"Saved {saved_count} SAM2 processed slices"
        
    except Exception as e:
        return False, None, f"Error saving results: {str(e)}"


def create_sam2_debug_images(save_dir, png_images, video_segments, bbox):
    """
    Create debug images for SAM2 post-processing.
    
    Args:
        save_dir: Save directory
        png_images: Original PNG images
        video_segments: SAM2 results
        bbox: Box prompt coordinates
        
    Returns:
        success: Boolean indicating success
        debug_dir: Debug directory path
    """
    try:
        # Create debug directory
        debug_dir = os.path.join(os.path.dirname(save_dir), 'sam_debug')
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save first slice with box
        if len(png_images) > 0:
            first_with_box = visualize_box_on_image(png_images[0], bbox)
            first_with_box.save(os.path.join(debug_dir, 'first_slice_with_box.png'))
        
        # Save sample SAM2 masks
        sample_count = 0
        for frame_idx in sorted(video_segments.keys()):
            if sample_count >= 5:  # Limit to 5 samples
                break
                
            if 1 in video_segments[frame_idx]:
                mask = video_segments[frame_idx][1]
                mask_img = (mask * 255).astype(np.uint8)
                Image.fromarray(mask_img, mode='L').save(
                    os.path.join(debug_dir, f'sam2_mask_frame_{frame_idx}.png')
                )
                sample_count += 1
        
        return True, debug_dir
        
    except Exception as e:
        print(f"Error creating debug images: {e}")
        return False, None


def cleanup_temp_folder(temp_dir):
    """Clean up temporary directory."""
    try:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory: {e}")
