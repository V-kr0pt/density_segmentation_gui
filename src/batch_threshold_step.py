import os
import io
import json
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from ImageLoader import UnifiedImageLoader
from utils import ImageOperations, ThresholdOperations
from new_utils import ThresholdOperator, ImageProcessor


# =============================
# Imports
# =============================

def batch_threshold_step():
    """
    Step 3: Batch threshold adjustment for all files.
    Organizes navigation, threshold selection, and saving for each file in the batch.
    """

    # =====================================================
    # UI header and styling
    # =====================================================
    st.header("🎯 Step 3: Adjust Thresholds")
    with open("static/batch_threshold_step.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # =====================================================
    # Session state and Data validation
    # =====================================================
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

    # =====================================================
    # Draw step completion check
    # =====================================================
    if len(completed_draw) < len(batch_files):
        st.warning(f"Please complete all drawing steps first. {len(completed_draw)}/{len(batch_files)} completed.")
        if st.button("← Back to Draw Step"):
            st.session_state["current_step"] = "batch_draw"
            st.rerun()
        return

    # =====================================================
    # Progress bar and overall navigation 
    # =====================================================
    total_files = len(batch_files)
    st.write(f"### Progress: {len(completed_threshold)}/{total_files} files completed")
    st.progress(len(completed_threshold) / total_files)
    st.markdown('</div>', unsafe_allow_html=True)

    # If all files are processed for threshold step
    if current_index >= len(batch_files):
        st.success("🎉 All thresholds have been set!")
        if st.button("→ Continue to Process Step"):
            st.session_state["current_step"] = "batch_process"
            st.rerun()
        if st.button("Back to Draw Step"):
            st.session_state["current_step"] = "batch_draw"
            st.rerun()
        return

    # =====================================================
    # File selection and skipping logic 
    # =====================================================
    current_file = batch_files[current_index]
    current_file_name = current_file.split('.')[0]

    # Skip if already completed: go to next pending file
    if current_file_name in completed_threshold:
        next_index = None
        for idx in range(current_index + 1, len(batch_files)):
            if batch_files[idx].split('.')[0] not in completed_threshold:
                next_index = idx
                break
        if next_index is not None:
            st.session_state["batch_current_index"] = next_index
        else:
            st.session_state["batch_current_index"] = len(batch_files)
        st.rerun()
        return

    # Skip if not in completed_draw: go to next file
    if current_file_name not in completed_draw:
        st.session_state["batch_current_index"] = current_index + 1
        st.rerun()
        return

    # =====================================================
    # Current file UI and navigation 
    # =====================================================
    st.markdown(f"""
    <div class="current-file">
        <h4>📁 {current_file} ({current_index + 1}/{total_files})</h4>
        <p>💡 Navigate between files to adjust thresholds • Saved files are locked</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if current_index > 0 and st.button("← Previous", help="Navigate to previous file"):
            st.session_state["batch_current_index"] = current_index - 1
            st.rerun()
    with col3:
        next_file_index = current_index + 1
        if next_file_index < len(batch_files) and st.button("Next →", help="Navigate to next file"):
            st.session_state["batch_current_index"] = next_file_index
            st.rerun()

    # =====================================================
    # Completed files list 
    # =====================================================
    if len(completed_threshold) > 0:
        with st.expander(f"✅ Completed Files ({len(completed_threshold)})", expanded=False):
            st.caption("*These files have saved thresholds and are locked*")
            cols = st.columns(3)
            for i, file in enumerate(completed_threshold):
                with cols[i % 3]:
                    st.markdown(f"🔒 **{file}.nii**")

    # =====================================================
    # File paths for current file 
    # =====================================================
    input_folder = os.path.join(os.getcwd(), 'media')
    output_path = os.path.join(os.getcwd(), 'output', current_file_name)
    mask_path = os.path.join(output_path, 'dense.nii')
    original_image_path = os.path.join(input_folder, current_file)

    # =====================================================
    # File existence and image loading 
    # =====================================================
    if not os.path.exists(mask_path):
        st.error(f"Mask file not found: {mask_path}")
        st.write("This file may not have completed the draw step properly.")
        if st.button("Skip this file"):
            st.session_state["batch_current_index"] = current_index + 1
            st.rerun()
        return

    try:
        img, _, _, _ = UnifiedImageLoader.load_slice(original_image_path)
        img_show = np.rot90(ImageProcessor.normalize_image(img)) # to show on GUI
        msk, _, _, _ = UnifiedImageLoader.load_slice(mask_path)
        msk_show = np.rot90(msk)
    except Exception as e:
        st.error(f"Error loading images: {str(e)}")
        if st.button("Skip this file"):
            st.session_state["batch_current_index"] = current_index + 1
            st.rerun()

    # =====================================================
    # Threshold preview and controls 
    # =====================================================
    st.markdown("### Threshold Configuration")
    
    # Main configuration panel
    with st.container():
        st.markdown('<div class="threshold-controls">', unsafe_allow_html=True)
        
        # Top row: Threshold control and view options
        config_col1, config_col2, config_col3 = st.columns([2, 2, 1])
        
        with config_col1:
            st.markdown("**Threshold Value**")
            threshold_key = f"threshold_number_{current_file_name}"
            saved_thresholds = st.session_state.get("batch_thresholds", {})
            default_threshold = saved_thresholds.get(current_file_name, 0.380)
            threshold = st.number_input(
                "Threshold",
                min_value=0.0,
                max_value=1.0,
                value=default_threshold,
                step=0.001,
                format="%.3f",
                key=threshold_key,
                help="Lower = more inclusive, Higher = more selective",
                label_visibility="collapsed"
            )
            
        with config_col2:
            st.markdown("**View Options**")
            comparison_mode = st.checkbox("Show side-by-side comparison",
                                        value=True, 
                                        key=f"comparison_{current_file_name}",
                                        help="Compare original image with thresholded result")
            
        with config_col3:
            st.markdown("**Image Width**")
            width_options = [400, 500, 600, 700, 800, 900, 1000, 1200, 1400]
            # Retrieve the last selected width index from session state or use default
            width_key = "last_selected_width_index"
            default_index = st.session_state.get(width_key, 4)
            
            selected_width = st.selectbox("Width", width_options, index=default_index, 
                                        key=f"width_{current_file}", 
                                        label_visibility="collapsed")
            
            # Save the selected index to session state for future use
            st.session_state[width_key] = width_options.index(selected_width)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Update session state
    if "batch_thresholds" not in st.session_state:
        st.session_state["batch_thresholds"] = {}
    st.session_state["batch_thresholds"][current_file_name] = threshold

    # Image preview section
    st.markdown("### Image Preview")
    
    if comparison_mode:
        # Side-by-side comparison
        img_col1, img_col2 = st.columns(2)
        
        with img_col1:
            st.markdown("**Original Image**")
            fig_raw, ax_raw = plt.subplots(figsize=(16, 16))
            ax_raw.imshow(img_show, cmap='gray')
            ax_raw.axis('off')
            buf_raw = io.BytesIO()
            fig_raw.savefig(buf_raw, format='png', bbox_inches='tight', pad_inches=0.05, 
                          facecolor='white', dpi=300)
            buf_raw.seek(0)
            plt.close(fig_raw)
            st.image(buf_raw, width=int(selected_width*0.60))
            
        with img_col2:
            st.markdown("**Thresholded Result**")
            # Display thresholded image
            fig_thresh = ThresholdOperator.display_thresholded_slice(img_show, msk_show, threshold)
            fig_thresh.set_size_inches(16, 16)
            buf_thresh = io.BytesIO()
            fig_thresh.savefig(buf_thresh, format='png', bbox_inches='tight', pad_inches=0.05, dpi=300)
            buf_thresh.seek(0)
            plt.close(fig_thresh)
            st.image(buf_thresh, width=int(selected_width*0.60))
            
    else:
        # Single thresholded image view
        # Center the image
        img_display_col1, img_display_col2, img_display_col3 = st.columns([1, 2, 1])
        with img_display_col2:
            fig = ThresholdOperator.display_thresholded_slice(img_show, msk_show, threshold)
            fig.set_size_inches(16, 16)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05, dpi=300)
            buf.seek(0)
            plt.close(fig)
            st.image(buf, width=selected_width)

    # Action buttons section
    st.markdown("---")
    button_col1, button_col2, button_col3 = st.columns([2, 1, 2])
    
    with button_col1:
        if st.button("Save & Continue", type="primary", use_container_width=True):
            if "batch_final_thresholds" not in st.session_state:
                st.session_state["batch_final_thresholds"] = {}
            st.session_state["batch_final_thresholds"][current_file_name] = threshold
            try:
                threshold_json = os.path.join(output_path, "threshold.json")
                with open(threshold_json, "w") as f:
                    json.dump({"threshold": threshold}, f)
                st.session_state["batch_completed_files"]["threshold"].append(current_file_name)
                st.session_state["batch_current_index"] = current_index + 1
                st.success(f"Threshold {threshold:.3f} saved successfully!")
            except Exception as e:
                st.error(f"Could not save threshold: {e}")
            st.rerun()
    
    with button_col3:
        if st.button("Back to Draw Step", use_container_width=True):
            st.session_state["current_step"] = "batch_draw"
            st.rerun()

    # =====================================================
    # Show current thresholds table 
    # =====================================================
    if "batch_final_thresholds" in st.session_state and len(st.session_state["batch_final_thresholds"]) > 0:
        with st.expander(f"📊 Saved Thresholds ({len(st.session_state['batch_final_thresholds'])})", expanded=False):
            cols = st.columns(3)
            for i, (file_name, thresh) in enumerate(st.session_state["batch_final_thresholds"].items()):
                with cols[i % 3]:
                    st.metric(f"{file_name}", f"{thresh:.3f}")
