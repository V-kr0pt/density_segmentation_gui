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
        # Import here to avoid circular imports
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        
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


def validate_sam2_results_quality(video_segments, original_png_count, 
                                 min_success_rate=0.5, visualization_placeholder=None):
    """
    Validate the quality of SAM2 video propagation results
    
    Args:
        video_segments: SAM2 propagation results
        original_png_count: Number of input PNG files
        min_success_rate: Minimum acceptable success rate (0.0-1.0)
        visualization_placeholder: Streamlit placeholder for visualizations
        
    Returns:
        tuple: (is_valid, quality_report, suggestions)
    """
    try:
        frame_count = len(video_segments)
        active_masks = sum(1 for frame_id in video_segments.keys() 
                         if 1 in video_segments[frame_id] and np.sum(video_segments[frame_id][1]) > 0)
        
        success_rate = active_masks / original_png_count if original_png_count > 0 else 0
        
        # Analyze mask quality
        mask_sizes = []
        mask_coverages = []
        
        for frame_id in video_segments.keys():
            if 1 in video_segments[frame_id]:
                mask = video_segments[frame_id][1]
                mask_size = np.sum(mask > 0)
                mask_coverage = mask_size / mask.size if mask.size > 0 else 0
                
                mask_sizes.append(mask_size)
                mask_coverages.append(mask_coverage)
        
        # Quality metrics
        avg_coverage = np.mean(mask_coverages) if mask_coverages else 0
        coverage_std = np.std(mask_coverages) if mask_coverages else 0
        
        # Generate report
        quality_report = {
            "success_rate": success_rate,
            "active_masks": active_masks,
            "total_frames": frame_count,
            "avg_coverage": avg_coverage,
            "coverage_consistency": 1.0 - coverage_std if coverage_std < 1.0 else 0.0,
            "mask_sizes": mask_sizes,
            "mask_coverages": mask_coverages
        }
        
        # Determine quality level
        is_valid = success_rate >= min_success_rate and avg_coverage > 0.01
        
        # Generate suggestions
        suggestions = []
        
        if success_rate < 0.7:
            suggestions.append("Low success rate - consider adjusting threshold parameters in batch_process_step")
            
        if avg_coverage < 0.05:
            suggestions.append("Small mask coverage - input PNGs may have insufficient segmentation")
            
        if coverage_std > 0.3:
            suggestions.append("Inconsistent mask sizes - check threshold adaptation quality")
            
        if active_masks == 0:
            suggestions.append("No active masks - check SAM2 box prompt configuration and input quality")
        
        # Display validation results
        if visualization_placeholder:
            with visualization_placeholder.container():
                st.markdown("### 🔍 **SAM2 Results Quality Analysis**")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Success Rate", f"{success_rate:.1%}")
                with col2:
                    st.metric("Avg Coverage", f"{avg_coverage:.1%}")
                with col3:
                    st.metric("Consistency", f"{quality_report['coverage_consistency']:.1%}")
                with col4:
                    if is_valid:
                        st.metric("Quality", "✅ Valid")
                    else:
                        st.metric("Quality", "⚠️ Poor")
                
                if suggestions:
                    st.markdown("**💡 Suggestions for improvement:**")
                    for suggestion in suggestions:
                        st.write(f"• {suggestion}")
                
                # Coverage distribution
                if mask_coverages:
                    st.markdown("**📊 Mask Coverage Distribution:**")
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.hist(mask_coverages, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.set_xlabel('Coverage Percentage')
                    ax.set_ylabel('Number of Frames')
                    ax.set_title('SAM2 Mask Coverage Distribution')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
        
        return is_valid, quality_report, suggestions
        
    except Exception as e:
        return False, {"error": str(e)}, [f"Error in quality validation: {str(e)}"]


def create_before_after_comparison(original_pngs, refined_segments, 
                                  visualization_placeholder=None, max_samples=3):
    """
    Create before/after comparison visualization showing batch_process_step vs SAM2 results
    
    Args:
        original_pngs: List of original PNG images from batch_process_step
        refined_segments: SAM2 video propagation results
        visualization_placeholder: Streamlit placeholder
        max_samples: Maximum number of samples to show
        
    Returns:
        bool: Success status
    """
    try:
        if not visualization_placeholder or not original_pngs or not refined_segments:
            return False
            
        # Select sample frames for comparison
        available_frames = list(refined_segments.keys())
        sample_indices = np.linspace(0, len(original_pngs)-1, 
                                   min(max_samples, len(original_pngs))).astype(int)
        
        with visualization_placeholder.container():
            st.markdown("### 🔄 **Before vs After: batch_process_step → SAM2**")
            st.markdown("*Comparison showing refinement achieved by SAM2 post-processing*")
            
            for i, idx in enumerate(sample_indices):
                if idx < len(original_pngs) and idx in available_frames:
                    st.markdown(f"**Sample {i+1}: Slice {idx}**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    # Original from batch_process_step
                    with col1:
                        st.markdown("**📥 batch_process_step**")
                        original_img = original_pngs[idx]
                        st.image(original_img, caption="Threshold-based segmentation", use_column_width=True)
                        
                        # Stats for original
                        orig_pixels = np.sum(original_img > 0)
                        orig_coverage = (orig_pixels / original_img.size) * 100
                        st.caption(f"Coverage: {orig_coverage:.1f}%")
                    
                    # SAM2 refined
                    with col2:
                        st.markdown("**🎯 SAM2 Refined**")
                        if 1 in refined_segments[idx]:
                            refined_mask = refined_segments[idx][1]
                            refined_img = (refined_mask * 255).astype(np.uint8)
                            st.image(refined_img, caption="Video propagation refined", use_column_width=True)
                            
                            # Stats for refined
                            refined_pixels = np.sum(refined_mask > 0)
                            refined_coverage = (refined_pixels / refined_mask.size) * 100
                            st.caption(f"Coverage: {refined_coverage:.1f}%")
                        else:
                            st.write("No refined mask available")
                            st.caption("Coverage: 0.0%")
                    
                    # Overlay comparison
                    with col3:
                        st.markdown("**🔍 Overlay Comparison**")
                        try:
                            if 1 in refined_segments[idx]:
                                # Create overlay: original in red, refined in green
                                refined_mask = refined_segments[idx][1]
                                
                                # Normalize both to 0-1
                                orig_norm = (original_img > 0).astype(float)
                                refined_norm = (refined_mask > 0).astype(float)
                                
                                # Create RGB overlay
                                overlay = np.zeros((*orig_norm.shape, 3))
                                overlay[:, :, 0] = orig_norm  # Red for original
                                overlay[:, :, 1] = refined_norm  # Green for refined
                                
                                st.image(overlay, caption="Red: Original, Green: Refined", use_column_width=True)
                                
                                # Calculate improvement metrics
                                intersection = np.sum((orig_norm > 0) & (refined_norm > 0))
                                union = np.sum((orig_norm > 0) | (refined_norm > 0))
                                iou = intersection / union if union > 0 else 0
                                st.caption(f"IoU: {iou:.3f}")
                            else:
                                st.write("No comparison available")
                        except Exception as e:
                            st.write("Overlay creation failed")
                    
                    st.markdown("---")
            
            # Overall improvement summary
            st.markdown("### 📊 **Overall Improvement Summary**")
            
            # Calculate overall metrics
            total_improvements = 0
            total_comparisons = 0
            
            for idx in sample_indices:
                if idx < len(original_pngs) and idx in available_frames and 1 in refined_segments[idx]:
                    orig_coverage = np.sum(original_pngs[idx] > 0) / original_pngs[idx].size
                    refined_coverage = np.sum(refined_segments[idx][1] > 0) / refined_segments[idx][1].size
                    
                    if refined_coverage > orig_coverage * 0.8:  # At least 80% of original coverage maintained
                        total_improvements += 1
                    total_comparisons += 1
            
            if total_comparisons > 0:
                improvement_rate = (total_improvements / total_comparisons) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Samples Analyzed", total_comparisons)
                with col2:
                    st.metric("Successful Refinements", total_improvements)
                with col3:
                    st.metric("Success Rate", f"{improvement_rate:.1f}%")
                
                if improvement_rate >= 80:
                    st.success("🟢 **Excellent refinement quality!** SAM2 successfully improved the segmentation.")
                elif improvement_rate >= 60:
                    st.success("🟡 **Good refinement quality.** SAM2 provided meaningful improvements.")
                else:
                    st.warning("🔴 **Refinement needs review.** Consider adjusting parameters.")
            
            st.info("💡 **SAM2 Post-Processing Approach**: SAM2 takes the PNG results from batch_process_step and applies intelligent video propagation to create more consistent and refined segmentations across all slices.")
        
        return True
        
    except Exception as e:
        if visualization_placeholder:
            with visualization_placeholder.container():
                st.error(f"Error creating comparison: {str(e)}")
        return False


def cleanup_temp_folder(temp_dir):

    """Clean up temporary directory."""
    try:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory: {e}")
