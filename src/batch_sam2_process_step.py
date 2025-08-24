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
import time
from sam2_postprocess_helpers import (
    convert_png_to_jpeg_for_sam2, 
    initialize_sam2_video_predictor,
    apply_box_prompt_to_first_slice,
    run_sam2_propagation,
    visualize_box_on_image,
    save_sam2_results,
    create_sam2_debug_images,
    validate_sam2_results_quality,
    create_before_after_comparison,
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
    
    # Store start time for performance tracking
    start_time = time.time()
    if 'st' in globals() and hasattr(st, 'session_state'):
        st.session_state.sam2_start_time = start_time
    
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
        
        # Check if dense_mask directory exists
        if not os.path.exists(dense_mask_dir):
            safe_update_status(f"❌ No batch_process_step results found at: {dense_mask_dir}", "error")
            safe_update_status("💡 Please run batch_process_step first to generate PNG files", "info")
            return False, f"No batch_process_step results found. Please run batch_process_step first.\nExpected directory: {dense_mask_dir}", None
        
        # Find all PNG files from batch_process_step
        all_files = os.listdir(dense_mask_dir)
        png_files = [f for f in all_files if f.endswith('.png')]
        png_files.sort()  # Ensure proper slice order
        
        if len(png_files) == 0:
            safe_update_status(f"❌ No PNG files found in {dense_mask_dir}", "error")
            safe_update_status(f"📁 Found files: {all_files[:5]}...", "info")
            return False, f"No PNG files found in {dense_mask_dir}. Please run batch_process_step first.", None
        
        safe_update_status(f"✅ Found {len(png_files)} PNG files from batch_process_step")
        safe_update_status(f"📂 Source: {dense_mask_dir}")
        
        # Show source visualization
        if visualization_placeholder:
            with visualization_placeholder.container():
                st.markdown("### 📁 **SAM2 Post-Processing Mode**")
                st.success(f"Found {len(png_files)} PNG files from batch_process_step")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info("**Source Directory:**")
                    st.code(dense_mask_dir)
                with col2:
                    st.info("**File Range:**")
                    st.write(f"• First: `{png_files[0]}`")
                    st.write(f"• Last: `{png_files[-1]}`")
                    st.write(f"• Total: {len(png_files)} files")
        
        # Step 2: Load PNG images and convert to JPEG for SAM2
        safe_update_status("🔄 Converting PNG results to JPEG format for SAM2...")
        safe_update_progress(2, 8, "Converting to JPEG")
        
        jpeg_success, temp_video_dir, png_images = convert_png_to_jpeg_for_sam2(dense_mask_dir, png_files)
        
        if not jpeg_success:
            safe_update_status("❌ Failed to convert PNG files to JPEG format", "error")
            return False, "Failed to convert PNG files to JPEG format for SAM2", None
        
        safe_update_status(f"✅ Created JPEG folder with {len(png_images)} frames")
        safe_update_status(f"📁 Temp directory: {temp_video_dir}")
        
        # Validate converted images
        if len(png_images) != len(png_files):
            cleanup_temp_folder(temp_video_dir)
            return False, f"Conversion mismatch: {len(png_files)} PNG files → {len(png_images)} converted", None
        
        # Show conversion visualization
        if visualization_placeholder and len(png_images) > 0:
            with visualization_placeholder.container():
                st.markdown("### 🖼️ **PNG → JPEG Conversion**")
                
                # Show sample images
                col1, col2, col3 = st.columns(3)
                sample_indices = [0, len(png_images)//2, -1]
                titles = ["First Slice", "Middle Slice", "Last Slice"]
                
                for i, (idx, title) in enumerate(zip(sample_indices, titles)):
                    if idx < len(png_images):
                        with [col1, col2, col3][i]:
                            st.markdown(f"**{title}**")
                            st.image(png_images[idx], caption=f"Slice {idx}", use_column_width=True)
                
                st.success(f"✅ Successfully converted {len(png_images)} PNG images to JPEG format for SAM2")
                st.info(f"📊 **Image Stats:** {png_images[0].shape[1]}×{png_images[0].shape[0]} pixels")
        
        # Step 3: Initialize SAM2 video predictor
        safe_update_status("🤖 Initializing SAM2 video predictor...")
        safe_update_progress(3, 8, "Loading SAM2")
        
        sam2_success, sam2_video, sam2_message = initialize_sam2_video_predictor(temp_video_dir)
        
        if not sam2_success:
            cleanup_temp_folder(temp_video_dir)
            safe_update_status(f"❌ SAM2 initialization failed: {sam2_message}", "error")
            return False, f"Failed to initialize SAM2: {sam2_message}", None
        
        safe_update_status(f"✅ SAM2 initialized successfully")
        safe_update_status(f"📋 {sam2_message}")
        
        # Step 4: Apply box prompt to first slice
        safe_update_status("🎯 Applying box prompt to first slice...")
        safe_update_progress(4, 8, "Box prompt")
        
        box_success, bbox, box_message = apply_box_prompt_to_first_slice(sam2_video, png_images[0])
        
        if not box_success:
            cleanup_temp_folder(temp_video_dir)
            safe_update_status(f"❌ Box prompt failed: {box_message}", "error")
            return False, f"Failed to apply box prompt: {box_message}", None
        
        safe_update_status(f"✅ Box prompt applied successfully")
        safe_update_status(f"📦 Box coordinates: {bbox}")
        
        # Show box prompt visualization
        if visualization_placeholder:
            with visualization_placeholder.container():
                st.markdown("### 🎯 **SAM2 Box Prompt Application**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📥 Input (from batch_process_step)**")
                    st.image(png_images[0], caption="First slice with thresholding applied", use_column_width=True)
                    
                    # Show statistics of first slice
                    non_zero_pixels = np.sum(png_images[0] > 0)
                    total_pixels = png_images[0].size
                    coverage = (non_zero_pixels / total_pixels) * 100
                    st.metric("Coverage", f"{coverage:.1f}%")
                
                with col2:
                    st.markdown("**📦 Box Prompt Applied**")
                    # Create visualization with box overlay
                    first_slice_with_box = visualize_box_on_image(png_images[0], bbox)
                    st.image(first_slice_with_box, caption="SAM2 box prompt region", use_column_width=True)
                    
                    # Show box details
                    x_min, y_min, x_max, y_max = bbox
                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    st.metric("Box Size", f"{int(box_width)}×{int(box_height)}")
                
                st.info(f"🎯 **Box prompt coordinates:** [{x_min:.0f}, {y_min:.0f}, {x_max:.0f}, {y_max:.0f}]")
        
        # Step 5: Run SAM2 video propagation
        safe_update_status("🔄 Running SAM2 video propagation...")
        safe_update_progress(5, 8, "Video propagation")
        
        prop_success, video_segments, prop_message = run_sam2_propagation(sam2_video)
        
        if not prop_success:
            cleanup_temp_folder(temp_video_dir)
            safe_update_status(f"❌ SAM2 propagation failed: {prop_message}", "error")
            return False, f"SAM2 propagation failed: {prop_message}", None
        
        safe_update_status(f"✅ Propagation completed successfully")
        safe_update_status(f"📊 Processed {len(video_segments)} frames")
        
        # Calculate statistics
        frame_count = len(video_segments)
        active_masks = sum(1 for frame_id in video_segments.keys() 
                         if 1 in video_segments[frame_id] and np.sum(video_segments[frame_id][1]) > 0)
        
        # Show propagation results
        if visualization_placeholder:
            with visualization_placeholder.container():
                st.markdown("### 🎬 **SAM2 Video Propagation Results**")
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Frames", frame_count)
                with col2:
                    st.metric("Active Masks", active_masks)
                with col3:
                    st.metric("Coverage", f"{(active_masks/frame_count)*100:.1f}%")
                with col4:
                    success_rate = (active_masks / frame_count) * 100
                    if success_rate >= 90:
                        st.metric("Quality", "🟢 Excellent")
                    elif success_rate >= 70:
                        st.metric("Quality", "🟡 Good") 
                    else:
                        st.metric("Quality", "🔴 Needs Review")
                
                # Show sample propagated masks
                if frame_count > 0:
                    st.markdown("**📊 Sample Propagated Masks:**")
                    sample_frames = sorted(list(video_segments.keys()))[:5]  # First 5 frames
                    
                    cols = st.columns(min(5, len(sample_frames)))
                    for i, frame_idx in enumerate(sample_frames):
                        if i < len(cols):
                            with cols[i]:
                                if 1 in video_segments[frame_idx]:
                                    mask = video_segments[frame_idx][1]
                                    mask_img = (mask * 255).astype(np.uint8)
                                    st.image(mask_img, caption=f"Frame {frame_idx}", use_column_width=True)
                                else:
                                    st.write(f"Frame {frame_idx}: No mask")
                
                if active_masks > 0:
                    st.success(f"✅ Successfully propagated masks to {active_masks} out of {frame_count} frames")
                else:
                    st.warning("⚠️ No active masks found after propagation - check input quality")
        
        # Step 5.5: Validate SAM2 results quality
        safe_update_status("🔍 Validating SAM2 results quality...")
        
        is_valid, quality_report, suggestions = validate_sam2_results_quality(
            video_segments, len(png_images), min_success_rate=0.3, 
            visualization_placeholder=visualization_placeholder
        )
        
        if not is_valid:
            safe_update_status("⚠️ Quality validation warnings found", "warning")
            if suggestions:
                for suggestion in suggestions:
                    safe_update_status(f"💡 {suggestion}", "info")
        else:
            safe_update_status("✅ Quality validation passed")
        
        # Step 6: Convert results back and save
        safe_update_status("💾 Converting results and saving...")
        safe_update_progress(6, 8, "Saving results")
        
        save_success, save_dir, save_message = save_sam2_results(
            nifti_path, output_dir, video_segments, png_images, 
            temp_video_dir, visualization_placeholder
        )
        
        if not save_success:
            cleanup_temp_folder(temp_video_dir)
            safe_update_status(f"❌ Failed to save results: {save_message}", "error")
            return False, f"Failed to save results: {save_message}", None
        
        safe_update_status(f"✅ Results saved successfully")
        
        # Show save results
        if visualization_placeholder:
            with visualization_placeholder.container():
                st.markdown("### 💾 **Results Saved**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**📁 Output Directory:**")
                    st.code(save_dir, language="text")
                    
                    # Count output files
                    if os.path.exists(save_dir):
                        png_count = len([f for f in os.listdir(save_dir) if f.endswith('.png')])
                        nii_count = len([f for f in os.listdir(save_dir) if f.endswith('.nii')])
                        st.metric("PNG Files", png_count)
                        st.metric("NIfTI Files", nii_count)
                
                with col2:
                    st.write("**📊 Processing Summary:**")
                    st.write(f"• Input PNGs from batch_process_step: {len(png_images)}")
                    st.write(f"• SAM2 propagated frames: {len(video_segments)}")
                    active_masks_count = sum(1 for frame_id in video_segments.keys() 
                                           if 1 in video_segments[frame_id] and np.sum(video_segments[frame_id][1]) > 0)
                    st.write(f"• Active refined masks: {active_masks_count}")
                    
                    success_rate = (active_masks_count / len(png_images)) * 100 if png_images else 0
                    if success_rate >= 90:
                        st.success(f"🟢 Excellent refinement: {success_rate:.1f}%")
                    elif success_rate >= 70:
                        st.success(f"🟡 Good refinement: {success_rate:.1f}%")
                    else:
                        st.warning(f"🔴 Needs review: {success_rate:.1f}%")
        
        # Step 7: Create debug visualizations and comparisons
        safe_update_status("📊 Creating debug visualizations and comparisons...")
        safe_update_progress(7, 8, "Debug images")
        
        debug_success, debug_dir = create_sam2_debug_images(
            save_dir, png_images, video_segments, bbox
        )
        
        if debug_success:
            safe_update_status(f"✅ Debug images saved: {debug_dir}")
        
        # Create before/after comparison showing batch_process_step → SAM2 refinement
        comparison_success = create_before_after_comparison(
            png_images, video_segments, visualization_placeholder, max_samples=3
        )
        
        if comparison_success:
            safe_update_status("✅ Before/after comparison created")
        
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
                st.markdown("### 🎉 **SAM2 Post-Processing Complete**")
                
                # Create comprehensive summary
                total_input = len(png_images)
                total_output = len(video_segments)
                active_masks = results['active_masks']
                success_rate = (active_masks / total_input) * 100 if total_input > 0 else 0
                
                # Main metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Input PNGs", total_input, help="From batch_process_step dense_mask folder")
                with col2:
                    st.metric("Propagated", total_output, help="SAM2 video propagation frames")
                with col3:
                    st.metric("Active Masks", active_masks, help="Frames with refined segmentation")
                with col4:
                    if success_rate >= 90:
                        st.metric("Quality", "🟢 Excellent", f"{success_rate:.1f}%")
                    elif success_rate >= 70:
                        st.metric("Quality", "🟡 Good", f"{success_rate:.1f}%")
                    else:
                        st.metric("Quality", "🔴 Review", f"{success_rate:.1f}%")
                
                # Success message
                if success_rate >= 90:
                    st.success(f"✅ **Excellent results!** SAM2 successfully refined {active_masks} out of {total_input} slices ({success_rate:.1f}% success rate)")
                elif success_rate >= 70:
                    st.success(f"✅ **Good results!** SAM2 refined {active_masks} out of {total_input} slices ({success_rate:.1f}% success rate)")
                elif success_rate > 0:
                    st.warning(f"⚠️ **Partial success.** SAM2 refined {active_masks} out of {total_input} slices ({success_rate:.1f}% success rate). Consider adjusting thresholds or input quality.")
                else:
                    st.error(f"❌ **No refinement achieved.** Check input PNG quality and SAM2 configuration.")
                
                # Output information
                st.markdown("### 📁 **Output Details**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🎯 SAM2 Refined Results:**")
                    st.code(save_dir, language="text")
                    st.caption("Contains refined PNG slices and NIfTI volume")
                    
                with col2:
                    if results.get('debug_dir'):
                        st.markdown("**🔍 Debug Visualizations:**")
                        st.code(results['debug_dir'], language="text")
                        st.caption("Contains overlay comparisons and analysis")
                    else:
                        st.markdown("**⚙️ Processing Details:**")
                        st.write(f"• Box prompt: {results.get('box_prompt', 'N/A')}")
                        st.write(f"• Video propagation: ✅ Complete")
                        st.write(f"• Post-processing approach: ✅ Using batch_process_step results")
                
                # Final note about the approach
                st.info("🔄 **SAM2 Post-Processing Approach**: This mode uses the PNG results from batch_process_step as input for SAM2 video propagation, providing intelligent refinement of the initial threshold-based segmentation.")
                
                # Show processing time if available
                if hasattr(st.session_state, 'sam2_start_time'):
                    processing_time = time.time() - st.session_state.sam2_start_time
                    st.caption(f"⏱️ Total processing time: {processing_time:.1f} seconds")
        
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


def batch_sam2_process_step():
    """
    Streamlit UI function for SAM2 post-processing mode
    
    This function provides the user interface for SAM2 video segmentation
    that post-processes results from batch_process_step.
    """
    st.markdown("## 🎬 SAM2 Video Segmentation (Post-Processing Mode)")
    st.markdown("""
    ### 🎯 **SAM2 Post-Processing Approach**
    
    This mode uses **SAM2 video propagation** to refine the segmentation results from the batch processing step.
    
    **Workflow:**
    1. 📁 **Input**: Uses PNG files from `dense_mask` folder (created by batch_process_step)
    2. 🎬 **SAM2 Processing**: Applies video propagation for temporal consistency  
    3. 💾 **Output**: Refined segmentation saved in `sam_mask` folder
    
    **Requirements:**
    - Run **Batch Process Step** first to generate PNG files
    - SAM2 model checkpoints must be available
    """)
    
    # Check if batch processing results exist
    if 'nifti_path' not in st.session_state or not st.session_state.nifti_path:
        st.warning("⚠️ Please select a NIfTI file in the File Selection step first.")
        return
    
    # Look for dense_mask folder
    nifti_dir = os.path.dirname(st.session_state.nifti_path)
    output_base_dir = os.path.join(nifti_dir, "output")
    dense_mask_dir = os.path.join(output_base_dir, "dense_mask")
    
    if not os.path.exists(dense_mask_dir):
        st.error("❌ **No batch processing results found!**")
        st.markdown("""
        Please run the **Batch Process Step** first to generate the PNG files that SAM2 will refine.
        
        Expected location: `{}/dense_mask/`
        """.format(output_base_dir))
        return
    
    # Check for PNG files
    png_files = [f for f in os.listdir(dense_mask_dir) if f.endswith('.png')]
    if not png_files:
        st.error("❌ **No PNG files found in dense_mask folder!**")
        st.markdown("Run the Batch Process Step to generate segmentation PNGs first.")
        return
    
    st.success(f"✅ **Found {len(png_files)} PNG files ready for SAM2 refinement**")
    st.info(f"📁 **Source**: `{dense_mask_dir}`")
    
    # SAM2 configuration
    st.markdown("### ⚙️ **SAM2 Configuration**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sam2_config = st.selectbox(
            "SAM2 Model Configuration",
            ["sam2_hiera_large.yaml", "sam2_hiera_base_plus.yaml", "sam2_hiera_small.yaml"],
            index=0,
            help="Choose SAM2 model configuration. Larger models are more accurate but slower."
        )
    
    with col2:
        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value="checkpoints/sam2_hiera_large.pt",
            help="Path to SAM2 model checkpoint file"
        )
    
    # Output directory
    sam_mask_dir = os.path.join(output_base_dir, "sam_mask")
    st.markdown(f"**📁 Output Directory**: `{sam_mask_dir}`")
    
    # Processing button
    if st.button("🚀 **Start SAM2 Post-Processing**", type="primary"):
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create visualization placeholder
        visualization_placeholder = st.empty()
        
        def update_progress(current, total, message=""):
            progress = current / total if total > 0 else 0
            progress_bar.progress(progress)
            if message:
                status_text.text(f"Step {current}/{total}: {message}")
        
        def update_status(message, level="info"):
            if level == "error":
                st.error(message)
            elif level == "warning":
                st.warning(message)
            elif level == "success":
                st.success(message)
            else:
                st.info(message)
        
        try:
            # Run SAM2 post-processing
            success, message, results = process_sam2_video_segmentation(
                nifti_path=st.session_state.nifti_path,
                mask_path=dense_mask_dir,
                threshold_data={},  # Not used in post-processing mode
                output_dir=sam_mask_dir,
                update_progress=update_progress,
                update_status=update_status,
                visualization_placeholder=visualization_placeholder
            )
            
            if success:
                st.success("🎉 **SAM2 Post-Processing Completed Successfully!**")
                
                if results:
                    st.markdown("### 📊 **Results Summary**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Input PNGs", results.get('source_pngs', 0))
                    with col2:
                        st.metric("Processed Frames", results.get('propagated_frames', 0))
                    with col3:
                        st.metric("Active Masks", results.get('active_masks', 0))
                    
                    st.markdown(f"**📁 Results saved to**: `{results.get('save_dir', sam_mask_dir)}`")
                    
                    if results.get('debug_dir'):
                        st.markdown(f"**🔍 Debug images**: `{results.get('debug_dir')}`")
            else:
                st.error(f"❌ **SAM2 Processing Failed**: {message}")
                
        except Exception as e:
            st.error(f"❌ **Error during SAM2 processing**: {str(e)}")
            
        finally:
            progress_bar.empty()
            status_text.empty()
