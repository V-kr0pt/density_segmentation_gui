import os
import io
import json
import streamlit as st
from utils import ImageOperations, ThresholdOperations

def batch_threshold_step():
    st.header("Step 2: Adjust Thresholds (Batch Mode)")
    
    # Check if we have batch data
    if "batch_files" not in st.session_state:
        st.error("No batch files selected. Please go back to file selection.")
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        return
    
    batch_files = st.session_state["batch_files"]
    current_index = st.session_state.get("batch_current_index", 0)
    completed_draw = st.session_state["batch_completed_files"]["draw"]
    all_completed_threshold = st.session_state["batch_completed_files"]["threshold"]
    completed_threshold = [f for f in all_completed_threshold if f in st.session_state["batch_files_without_extension"]] # only the set of batch files that already has a polygon
    
    # check if the threshold.json is in the output folder
    for file in completed_threshold:
        output_path = os.path.join(os.getcwd(), 'output', file)
        threshold_json = os.path.join(output_path, "threshold.json")
        if not os.path.exists(threshold_json):
            completed_threshold.remove(file)
        
    
    # Check if all draw steps are completed
    if len(completed_draw) < len(batch_files):
        st.warning(f"Please complete all drawing steps first. {len(completed_draw)}/{len(batch_files)} completed.")
        if st.button("← Back to Draw Step"):
            st.session_state["current_step"] = "batch_draw"
            st.rerun()
        return
    
    # Progress info
    total_files = len(batch_files)
    st.write(f"### Progress: {len(completed_threshold)}/{total_files} files completed")
    st.progress(len(completed_threshold) / total_files)
    
    # Find next file to process
    if current_index >= len(batch_files):
        # All files processed for threshold step
        st.success("🎉 All files have been processed for threshold step!")
        if st.button("➡️ Proceed to Process Step"):
            st.session_state["current_step"] = "batch_process"
            st.rerun()
        if st.button("← Back to Draw Step"):
            st.session_state["current_step"] = "batch_draw"
            st.rerun()
        return
    
    current_file = batch_files[current_index]
    current_file_name = current_file.split('.')[0]
    
    # Skip if already completed -> search the next pending
    if current_file_name in completed_threshold:
        next_index = None
        for idx in range(current_index + 1, len(batch_files)):
            if batch_files[idx].split('.')[0] not in completed_threshold:
                next_index = idx
                break
        if next_index is not None:
            st.session_state["batch_current_index"] = next_index
        else:
            # Nenhum pendente encontrado
            st.session_state["batch_current_index"] = len(batch_files)
        st.rerun()
        return
    
    # Skip if not in completed_draw
    if current_file_name not in completed_draw:
        st.session_state["batch_current_index"] = current_index + 1
        st.rerun()
        return
    
    st.write(f"### Currently processing: `{current_file}` ({current_index + 1}/{total_files})")
    
    # File navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if current_index > 0 and st.button("← Previous File"):
            st.session_state["batch_current_index"] = current_index - 1
            st.rerun()
    
    with col3:
        next_file_index = current_index + 1
        if next_file_index < len(batch_files) and st.button("Next File →"):
            st.session_state["batch_current_index"] = next_file_index
            st.rerun()
    
    # Show completed files list
    if len(completed_threshold) > 0:
        with st.expander(f"Completed files ({len(completed_threshold)})"):
            for file in completed_threshold:
                st.write(f"✅ {file}.nii")
    
    # Process current file
    input_folder = os.path.join(os.getcwd(), 'media')
    output_path = os.path.join(os.getcwd(), 'output', current_file_name)
    mask_path = os.path.join(output_path, 'dense.nii')
    original_image_path = os.path.join(input_folder, current_file)
    
    # Check if required files exist
    if not os.path.exists(mask_path):
        st.error(f"Mask file not found: {mask_path}")
        st.write("This file may not have completed the draw step properly.")
        if st.button("Skip this file"):
            st.session_state["batch_current_index"] = current_index + 1
            st.rerun()
        return
    
    try:
        img = ImageOperations.load_central_slice_any(original_image_path)
        msk = ImageOperations.load_nii_central_slice(mask_path, flip=True)
    except Exception as e:
        st.error(f"Error loading images: {str(e)}")
        if st.button("Skip this file"):
            st.session_state["batch_current_index"] = current_index + 1
            st.rerun()
        return
    
    # Display options
    width_options = [400, 500, 600, 700, 800, 900, 1000]
    selected_width = st.selectbox("Select image width", width_options, index=2, key=f"width_{current_file}")
    
    # Threshold slider
    threshold_key = f"threshold_slider_{current_file_name}"
    
    saved_thresholds = st.session_state.get("batch_thresholds", {})
    default_threshold = saved_thresholds.get(current_file_name, 0.38)
    
    threshold = st.slider(
        "Select threshold value",
        min_value=0.0,
        max_value=1.0,
        value=default_threshold,
        step=0.01,
        key=threshold_key
    )
    
    if "batch_thresholds" not in st.session_state:
        st.session_state["batch_thresholds"] = {}
    st.session_state["batch_thresholds"][current_file_name] = threshold
    
    # Display thresholded image
    fig = ThresholdOperations.display_thresholded_slice(img, msk, threshold)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    cols = st.columns([1, 1, 1])
    with cols[1]:
        st.image(buf, caption="Thresholded Central Slice", width=selected_width)
    
    # Save button
    if st.button("💾 Save Threshold and Continue to Next File"):
        # Save threshold in session
        if "batch_final_thresholds" not in st.session_state:
            st.session_state["batch_final_thresholds"] = {}
        st.session_state["batch_final_thresholds"][current_file_name] = threshold

        # Save threshold to disk
        try:
            threshold_json = os.path.join(output_path, "threshold.json")
            with open(threshold_json, "w") as f:
                json.dump({"threshold": threshold}, f)

             # Mark file as completed
            st.session_state["batch_completed_files"]["threshold"].append(current_file_name)

            # Move to next file
            st.session_state["batch_current_index"] = current_index + 1

            st.success(f"Threshold {threshold:.2f} saved for {current_file}!")
        
        except Exception as e:
            st.warning(f"Could not save threshold for {current_file_name}: {e}")
        
        st.rerun()      
    
    # Show current thresholds
    if "batch_final_thresholds" in st.session_state and len(st.session_state["batch_final_thresholds"]) > 0:
        with st.expander("Saved Thresholds"):
            for file_name, thresh in st.session_state["batch_final_thresholds"].items():
                st.write(f"📁 {file_name}.nii: {thresh:.3f}")
    
    # Back button
    if st.button("← Back to Draw Step"):
        st.session_state["current_step"] = "batch_draw"
        st.rerun()
