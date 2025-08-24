"""
SAM2 Post-Processing Mode

This module implements SAM2 as a post-processing step that takes the results
from batch_process_step (PNG files) and applies SAM2 video propagation for refinement.

Workflow:
1. Load existing PNG results from batch_process_step (dense_mask folder)
2. Convert PNGs to JPEG format for SAM2 video predictor
3. Apply box prompt to first slice 
4. Use SAM2 video propagation to refine all slices
5. Save refined results in sam_mask folder
"""

import os
import numpy as np
import nibabel
import streamlit as st
from sam2_postprocess_helpers import (
    convert_png_to_jpeg_for_sam2, 
    initialize_sam2_video_predictor,
    apply_box_prompt_to_first_slice,
    run_sam2_propagation,
    visualize_box_on_image,
    save_sam2_results,
    create_sam2_debug_images,
    cleanup_temp_folder
)


def process_sam2_video_segmentation(nifti_path, mask_path, threshold_data, output_dir, 
                                  update_progress=None, update_status=None, 
                                  visualization_placeholder=None):
    """
    SAM2 Video Segmentation Mode - Post-processes batch_process_step results
    
    This mode takes the PNG results from batch_process_step (which applies dynamic threshold)
    and uses SAM2 video propagation to refine the segmentation.
    
    Args:
        nifti_path: Path to NIfTI file
        mask_path: Path to mask file (if any)
        threshold_data: Threshold configuration (not used, since we use existing PNGs)
        output_dir: Output directory
        update_progress: Progress callback function
        update_status: Status callback function
        visualization_placeholder: Streamlit placeholder for visualizations
        
    Returns:
        success: Boolean indicating success
        message: Status message
        results: Dictionary with processing results
    """
    
    def safe_update_progress(current, total, message=""):
        if update_progress:
            try:
                update_progress(current, total, message)
            except:
                pass
    
    def safe_update_status(message, status="info"):
        if update_status:
            try:
                update_status(message, status)
            except:
                pass
        print(f"[SAM2] {message}")
    
    try:
        safe_update_status("🚀 Starting SAM2 post-processing mode...")
        
        # Step 1: Find existing batch_process_step results
        safe_update_status("📁 Looking for existing batch_process_step results...")
        safe_update_progress(1, 8, "Finding PNG results")
        
        filename_no_ext = os.path.splitext(os.path.basename(nifti_path))[0]
        source_output_path = os.path.join(output_dir, filename_no_ext)
        dense_mask_dir = os.path.join(source_output_path, 'dense_mask')
        
        if not os.path.exists(dense_mask_dir):
            return False, f"No batch_process_step results found. Please run batch_process_step first. Expected: {dense_mask_dir}", None
        
        # Find all PNG files from batch_process_step
        png_files = [f for f in os.listdir(dense_mask_dir) if f.endswith('.png')]
        png_files.sort()  # Ensure proper slice order
        
        if len(png_files) == 0:
            return False, f"No PNG files found in {dense_mask_dir}. Please run batch_process_step first.", None
        
        safe_update_status(f"✅ Found {len(png_files)} PNG files from batch_process_step")
        
        # Show source visualization
        if visualization_placeholder:
            with visualization_placeholder.container():
                st.markdown("### 📁 **SAM2 Post-Processing Mode**")
                st.info(f"Using {len(png_files)} PNG files from batch_process_step as input")
                st.write(f"**Source:** `{dense_mask_dir}`")
                st.write(f"**Files:** {png_files[0]} → {png_files[-1]}")
        
        # Step 2: Load PNG images and convert to JPEG for SAM2
        safe_update_status("🔄 Converting PNG results to JPEG format for SAM2...")
        safe_update_progress(2, 8, "Converting to JPEG")
        
        jpeg_success, temp_video_dir, png_images = convert_png_to_jpeg_for_sam2(dense_mask_dir, png_files)
        
        if not jpeg_success:
            return False, "Failed to convert PNG files to JPEG format for SAM2", None
        
        safe_update_status(f"✅ Created JPEG folder: {temp_video_dir}")
        
        # Show conversion visualization
        if visualization_placeholder and len(png_images) > 0:
            with visualization_placeholder.container():
                st.markdown("### 🖼️ **PNG → JPEG Conversion**")
                
                # Show sample images
                col1, col2, col3 = st.columns(3)
                sample_indices = [0, len(png_images)//2, -1]
                
                for i, idx in enumerate(sample_indices):
                    if idx < len(png_images):
                        with [col1, col2, col3][i]:
                            st.image(png_images[idx], caption=f"Slice {idx}", use_column_width=True)
                
                st.success(f"✅ Converted {len(png_images)} PNG images to JPEG format")
        
        # Step 3: Initialize SAM2 video predictor
        safe_update_status("🤖 Initializing SAM2 video predictor...")
        safe_update_progress(3, 8, "Loading SAM2")
        
        sam2_success, sam2_video, sam2_message = initialize_sam2_video_predictor(temp_video_dir)
        
        if not sam2_success:
            cleanup_temp_folder(temp_video_dir)
            return False, f"Failed to initialize SAM2: {sam2_message}", None
        
        safe_update_status(f"✅ SAM2 initialized: {sam2_message}")
        
        # Step 4: Apply box prompt to first slice
        safe_update_status("🎯 Applying box prompt to first slice...")
        safe_update_progress(4, 8, "Box prompt")
        
        box_success, bbox, box_message = apply_box_prompt_to_first_slice(sam2_video, png_images[0])
        
        if not box_success:
            cleanup_temp_folder(temp_video_dir)
            return False, f"Failed to apply box prompt: {box_message}", None
        
        safe_update_status(f"✅ Box prompt applied: {box_message}")
        
        # Show box prompt visualization
        if visualization_placeholder:
            with visualization_placeholder.container():
                st.markdown("### 🎯 **SAM2 Box Prompt (First Slice)**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original First Slice**")
                    st.image(png_images[0], caption="Input from batch_process_step", use_column_width=True)
                
                with col2:
                    st.markdown("**Box Prompt Applied**")
                    # Create visualization with box overlay
                    first_slice_with_box = visualize_box_on_image(png_images[0], bbox)
                    st.image(first_slice_with_box, caption=f"Box: {bbox}", use_column_width=True)
                
                st.info(f"📦 **Box coordinates:** {bbox}")
        
        # Step 5: Run SAM2 video propagation
        safe_update_status("🔄 Running SAM2 video propagation...")
        safe_update_progress(5, 8, "Video propagation")
        
        prop_success, video_segments, prop_message = run_sam2_propagation(sam2_video)
        
        if not prop_success:
            cleanup_temp_folder(temp_video_dir)
            return False, f"SAM2 propagation failed: {prop_message}", None
        
        safe_update_status(f"✅ Propagation complete: {prop_message}")
        
        # Show propagation results
        if visualization_placeholder:
            with visualization_placeholder.container():
                st.markdown("### 🎬 **SAM2 Video Propagation Results**")
                
                # Statistics
                frame_count = len(video_segments)
                active_masks = sum(1 for frame_id in video_segments.keys() 
                                 if 1 in video_segments[frame_id] and np.sum(video_segments[frame_id][1]) > 0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Frames", frame_count)
                with col2:
                    st.metric("Active Masks", active_masks)
                with col3:
                    st.metric("Coverage", f"{(active_masks/frame_count)*100:.1f}%")
        
        # Step 6: Convert results back and save
        safe_update_status("💾 Converting results and saving...")
        safe_update_progress(6, 8, "Saving results")
        
        save_success, save_dir, save_message = save_sam2_results(
            nifti_path, output_dir, video_segments, png_images, 
            temp_video_dir, visualization_placeholder
        )
        
        if not save_success:
            cleanup_temp_folder(temp_video_dir)
            return False, f"Failed to save results: {save_message}", None
        
        safe_update_status(f"✅ Results saved: {save_message}")
        
        # Step 7: Create debug visualizations
        safe_update_status("📊 Creating debug visualizations...")
        safe_update_progress(7, 8, "Debug images")
        
        debug_success, debug_dir = create_sam2_debug_images(
            save_dir, png_images, video_segments, bbox
        )
        
        if debug_success:
            safe_update_status(f"✅ Debug images saved: {debug_dir}")
        
        # Final cleanup
        safe_update_progress(8, 8, "Cleanup")
        cleanup_temp_folder(temp_video_dir)
        
        # Prepare results
        results = {
            "save_dir": save_dir,
            "debug_dir": debug_dir if debug_success else None,
            "source_pngs": len(png_images),
            "propagated_frames": len(video_segments),
            "active_masks": active_masks if 'active_masks' in locals() else 0,
            "box_prompt": bbox.tolist() if isinstance(bbox, np.ndarray) else bbox
        }
        
        safe_update_status("🎉 SAM2 post-processing completed successfully!", "success")
        
        # Final summary visualization
        if visualization_placeholder:
            with visualization_placeholder.container():
                st.markdown("### ✅ **SAM2 Post-Processing Complete**")
                st.success(f"Successfully processed {len(png_images)} slices using SAM2 video propagation")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**📁 Output Location:**")
                    st.code(save_dir)
                with col2:
                    st.write("**📊 Processing Stats:**")
                    st.write(f"• Source PNGs: {len(png_images)}")
                    st.write(f"• Propagated: {len(video_segments)}")
                    st.write(f"• Active masks: {results['active_masks']}")
        
        return True, "SAM2 post-processing completed successfully", results
        
    except Exception as e:
        if 'temp_video_dir' in locals():
            cleanup_temp_folder(temp_video_dir)
        safe_update_status(f"❌ Error in SAM2 processing: {str(e)}", "error")
        return False, f"Error processing with SAM2: {str(e)}", None


# Legacy function names for compatibility
def process_sam2_batch_step(nifti_path, mask_path, threshold_data, output_dir, 
                           update_progress=None, update_status=None, 
                           visualization_placeholder=None):
    """Legacy function name - redirects to new implementation"""
    return process_sam2_video_segmentation(
        nifti_path, mask_path, threshold_data, output_dir,
        update_progress, update_status, visualization_placeholder
    )
