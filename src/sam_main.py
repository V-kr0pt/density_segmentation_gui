"""
SAM2 Main Processing Steps
"""
import os
import streamlit as st
from utils import ImageOperations
from sam_utils import SAM2Manager, show_sam2_setup_info

def sam_step():
    """SAM2 file selection step - similar to file_selection_step but for single file"""
    st.header("SAM2 Processing - File Selection")
    
    input_folder = os.path.join(os.getcwd(), 'media')
    
    if not os.path.exists(input_folder):
        st.error(f"Media folder not found: {input_folder}")
        return
    
    # Get available .nii files
    available_files = [f for f in os.listdir(input_folder) if f.endswith('.nii')]
    
    if not available_files:
        st.warning("No .nii files found in the media folder.")
        return
    
    st.write("### Select a .nii file for SAM2 processing:")
    
    # File selection
    selected_file = st.selectbox("Choose file:", available_files, key="sam_file_selection")
    
    if selected_file:
        st.write(f"Selected file: `{selected_file}`")
        
        # Preview central slice
        file_path = os.path.join(input_folder, selected_file)
        try:
            preview_image = ImageOperations.load_nii_central_slice(file_path)
            
            # Normalize image to [0, 1] range for display
            normalized_preview = (preview_image - preview_image.min()) / (preview_image.max() - preview_image.min())
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(normalized_preview, caption=f"Central slice of {selected_file}", use_container_width=True, clamp=True)
            
        except Exception as e:
            st.error(f"Error loading preview: {str(e)}")
    
    # Check SAM2 setup
    with st.expander("SAM2 Setup Status"):
        sam_manager = SAM2Manager()
        
        # Check dependencies
        deps_ok, deps_msg = sam_manager.check_dependencies()
        if deps_ok:
            st.success(f"✅ Dependencies: {deps_msg}")
        else:
            st.error(f"❌ Dependencies: {deps_msg}")
            show_sam2_setup_info()
            return
        
        # Check checkpoint
        checkpoint_ok, checkpoint_msg = sam_manager.check_checkpoint()
        if checkpoint_ok:
            st.success(f"✅ Checkpoint: {checkpoint_msg}")
        else:
            st.error(f"❌ Checkpoint: {checkpoint_msg}")
            st.info("Download SAM2.1 checkpoint from: https://github.com/facebookresearch/segment-anything-2")
            return
    
    # Proceed button
    if st.button("🚀 Start SAM2 Processing", type="primary"):
        if selected_file:
            # Store selected file info
            st.session_state["sam_selected_file"] = selected_file
            st.session_state["sam_file_path"] = os.path.join(input_folder, selected_file)
            st.session_state["sam_file_name"] = selected_file.split('.')[0]
            
            # Move to draw step (similar to batch process)
            st.session_state["current_step"] = "sam_draw"
            st.rerun()
        else:
            st.warning("Please select a file first.")

def sam_threshold_step():
    """Manual threshold step for SAM2 - similar to batch_threshold_step but for single file"""
    st.header("SAM2 Processing - Step 2: Adjust Threshold")
    
    if "sam_selected_file" not in st.session_state:
        st.error("No file selected. Please go back to file selection.")
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "sam"
            st.rerun()
        return
    
    selected_file = st.session_state["sam_selected_file"]
    file_path = st.session_state["sam_file_path"]
    file_name = st.session_state["sam_file_name"]
    
    st.write(f"### Processing threshold for: `{selected_file}`")
    
    # Process current file
    input_folder = os.path.join(os.getcwd(), 'media')
    output_path = os.path.join(os.getcwd(), 'output', file_name)
    mask_path = os.path.join(output_path, 'dense.nii')
    original_image_path = file_path
    
    # Check if required files exist
    if not os.path.exists(mask_path):
        st.error(f"Mask file not found: {mask_path}")
        st.write("This file may not have completed the draw step properly.")
        if st.button("← Back to Draw Step"):
            st.session_state["current_step"] = "sam_draw"
            st.rerun()
        return
    
    try:
        from utils import ImageOperations, ThresholdOperations
        import io
        import json
        
        img = ImageOperations.load_nii_central_slice(original_image_path)
        msk = ImageOperations.load_nii_central_slice(mask_path, flip=True)
    except Exception as e:
        st.error(f"Error loading images: {str(e)}")
        if st.button("← Back to Draw Step"):
            st.session_state["current_step"] = "sam_draw"
            st.rerun()
        return
    
    # Display options
    width_options = [400, 500, 600, 700, 800, 900, 1000]
    selected_width = st.selectbox("Select image width", width_options, index=2, key="sam_width")
    
    # Threshold slider
    threshold_key = "sam_threshold_slider"
    
    # Check if threshold already saved
    threshold_json = os.path.join(output_path, "threshold.json")
    default_threshold = 0.38
    if os.path.exists(threshold_json):
        try:
            with open(threshold_json, "r") as f:
                data = json.load(f)
                default_threshold = data.get("threshold", 0.38)
        except:
            pass
    
    threshold = st.slider(
        "Select threshold value",
        min_value=0.0,
        max_value=1.0,
        value=default_threshold,
        step=0.01,
        key=threshold_key
    )
    
    # Display thresholded image
    fig = ThresholdOperations.display_thresholded_slice(img, msk, threshold)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    cols = st.columns([1, 1, 1])
    with cols[1]:
        st.image(buf, caption="Thresholded Central Slice", width=selected_width)
    
    # Save button
    if st.button("💾 Save Threshold and Continue to Auto Processing"):
        # Save threshold to disk
        try:
            os.makedirs(output_path, exist_ok=True)
            with open(threshold_json, "w") as f:
                json.dump({"threshold": threshold}, f)

            st.session_state["sam_final_threshold"] = threshold
            st.success(f"Threshold {threshold:.2f} saved for {selected_file}!")
            
            # Move to auto threshold detection for SAM2
            st.session_state["current_step"] = "sam_threshold_auto"
            st.rerun()
        
        except Exception as e:
            st.error(f"Could not save threshold for {file_name}: {e}")
    
    # Back button
    if st.button("← Back to Draw Step"):
        st.session_state["current_step"] = "sam_draw"
        st.rerun()

def sam_process_step():
    """Final processing step - save results"""
    st.header("SAM2 Processing - Final Results")
    
    if "sam_video_segments" not in st.session_state:
        st.error("No SAM2 results available. Please complete the propagation step first.")
        if st.button("← Back to Propagation"):
            st.session_state["current_step"] = "sam_propagation"
            st.rerun()
        return
    
    st.success("🎉 SAM2 processing completed!")
    
    # Display summary
    video_segments = st.session_state["sam_video_segments"]
    file_name = st.session_state["sam_file_name"]
    
    st.write(f"**Processed file:** {st.session_state['sam_selected_file']}")
    st.write(f"**Total frames processed:** {len(video_segments)}")
    
    # Show some results
    st.subheader("Processing Summary")
    for frame_idx, masks in list(video_segments.items())[:5]:  # Show first 5 frames
        st.write(f"Frame {frame_idx}: {len(masks)} objects detected")
    
    if len(video_segments) > 5:
        st.write(f"... and {len(video_segments) - 5} more frames")
    
    # Save results button
    if st.button("💾 Save Results"):
        try:
            output_path = os.path.join(os.getcwd(), 'output', file_name + '_sam2')
            os.makedirs(output_path, exist_ok=True)
            
            # Here you would save the actual segmentation results
            # This is a placeholder - implement actual saving logic
            
            import json
            results_summary = {
                "file": st.session_state["sam_selected_file"],
                "total_frames": len(video_segments),
                "processing_method": "SAM2",
                "threshold": st.session_state.get("sam_threshold", 0.45)
            }
            
            with open(os.path.join(output_path, "sam2_results.json"), "w") as f:
                json.dump(results_summary, f, indent=2)
            
            st.success(f"Results saved to: {output_path}")
            
        except Exception as e:
            st.error(f"Error saving results: {str(e)}")
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Process Another File"):
            # Clear SAM2 session data
            keys_to_clear = [k for k in st.session_state.keys() if k.startswith("sam_")]
            for key in keys_to_clear:
                del st.session_state[key]
            st.session_state["current_step"] = "sam"
            st.rerun()
    
    with col2:
        if st.button("← Back to Propagation"):
            st.session_state["current_step"] = "sam_propagation"
            st.rerun()
