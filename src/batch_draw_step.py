import os
import streamlit as st
from PIL import Image
from utils import ImageOperations, MaskOperations
from streamlit_drawable_canvas import st_canvas
import numpy as np

def batch_draw_step():
    st.header("🎨 Step 2: Draw Masks")
    
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
    .step-container h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.2rem;
        color: #2d3436;
    }
    .step-container p {
        margin: 0;
        font-size: 0.9rem;
        color: #2d3436;
    }
    .progress-section {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        box-shadow: 0 1px 5px rgba(0,0,0,0.05);
    }
    .progress-section h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
    }
    .current-file {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2d3436;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(168, 237, 234, 0.3);
        border-left: 3px solid #00b894;
    }
    .current-file h4 {
        margin: 0 0 0.3rem 0;
        font-size: 1.1rem;
        color: #2d3436;
    }
    .current-file p {
        margin: 0;
        font-size: 0.9rem;
        color: #2d3436;
    }
    .canvas-container {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f0f0f0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .canvas-container h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
    }
    .compact-section {
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    
    # Check if we have batch data
    if "batch_files" not in st.session_state:
        st.error("No batch files selected. Please go back to file selection.")
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        return
    
    batch_files = st.session_state["batch_files"]
    current_index = st.session_state.get("batch_current_index", 0)
    all_completed_draw = st.session_state["batch_completed_files"]["draw"] # return all files that already has a polygon 
    completed_draw = [f for f in all_completed_draw if f in st.session_state["batch_files_without_extension"]] # only the set of batch files that already has a polygon
    
    # Progress info
    #st.markdown('<div class="progress-section">', unsafe_allow_html=True)
    total_files = len(batch_files)
    st.write(f"### Progress: {len(completed_draw)}/{total_files} files completed")
    st.progress(len(completed_draw) / total_files)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Find next file to process
    if current_index >= len(batch_files):
        # All files processed for draw step
        st.success("🎉 All masks have been drawn!")
        if st.button("→ Continue to Threshold Step"):
            st.session_state["batch_current_index"] = 0
            st.session_state["current_step"] = "batch_threshold"
            st.rerun()
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        return
    
    current_file = batch_files[current_index]
    current_file_name = current_file.split('.')[0]
    
    # Skip if already completed
    if current_file_name in completed_draw:
        st.session_state["batch_current_index"] = current_index + 1
        st.rerun()
        return
    
    st.markdown(f"""
    <div class="current-file">
        <h4>📁 {current_file} ({current_index + 1}/{total_files})</h4>
        <p>💡 Navigate between unsaved files • Saved files are locked</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if current_index > 0 and st.button("← Previous", help="Navigate to previous unsaved file"):
            st.session_state["batch_current_index"] = current_index - 1
            # Clear current file session state
            keys_to_clear = ["points", "mask", "result", "create_mask", "affine", "original_image_path", "nb_of_slices", "scale", "output_path"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    with col3:
        next_file_index = current_index + 1
        if next_file_index < len(batch_files) and st.button("Next →", help="Navigate to next unsaved file"):
            st.session_state["batch_current_index"] = next_file_index
            # Clear current file session state
            keys_to_clear = ["points", "mask", "result", "create_mask", "affine", "original_image_path", "nb_of_slices", "scale", "output_path"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Show completed files list
    if len(completed_draw) > 0:
        with st.expander(f"✅ Completed Files ({len(completed_draw)})", expanded=False):
            st.caption("*These files are saved and locked*")
            cols = st.columns(3)
            for i, file in enumerate(completed_draw):
                with cols[i % 3]:
                    st.markdown(f"🔒 **{file}.nii**")
    
    st.divider()
    
    # Process current file
    input_folder = os.path.join(os.getcwd(), 'media')
    file_path = os.path.join(input_folder, current_file)
    
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return
    
    # Load image (same logic as original draw_step)
    image, affine, nb_of_slices = ImageOperations.load_image(file_path)
    
    if "affine" not in st.session_state or st.session_state.get("current_batch_file") != current_file:
        st.session_state["affine"] = affine
        st.session_state["original_image_path"] = file_path
        st.session_state["nb_of_slices"] = nb_of_slices
        st.session_state["current_batch_file"] = current_file
        st.session_state["points"] = []  # Reset points for new file
        # Clear other file-specific data
        for key in ["mask", "result", "create_mask"]:
            if key in st.session_state:
                del st.session_state[key]
    
    max_width, max_height = 1200, 800
    orig_height, orig_width = image.shape[0], image.shape[1]
    scale = min(max_width / orig_width, max_height / orig_height, 1)
    
    if "scale" not in st.session_state:
        st.session_state["scale"] = scale
    
    pil_width = int(orig_width * scale)
    pil_height = int(orig_height * scale)
    pil_image = Image.fromarray(image).resize((pil_width, pil_height)).rotate(90, expand=True)
    
    if "points" not in st.session_state:
        st.session_state.points = []
    
    # Drawing canvas
    #st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
    st.markdown("### 🖌️ Drawing Canvas")
    
    # Adjust canvas size for rotated image
    rotated_width, rotated_height = pil_image.size
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="red",
        background_image=pil_image,
        update_streamlit=True,
        height=rotated_height,
        width=rotated_width,
        drawing_mode="polygon",
        key=f"canvas_{current_file}"  # Unique key per file
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Control buttons in a single row
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎨 Create Mask"):
            st.session_state["create_mask"] = True
    with col2:
        if st.button("🗑️ Clear Drawing"):
            keys_to_clear = ["points", "mask", "result", "create_mask"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    with col3:
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
    
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if len(objects) > 0:
            polygon = objects[-1]["path"]
            # Transform points from rotated canvas back to original orientation
            points_rotated = [(int(p[1]), int(p[2])) for p in polygon if len(p) == 3]
            # For 90 degree rotation (counterclockwise), new_x = y, new_y = width - x
            points = [(rotated_height - y, x) for x, y in points_rotated]
            st.session_state.points = points
    
    if st.session_state.get("create_mask", False):
        if len(st.session_state.points) >= 3:
            result, mask = MaskOperations.create_mask(image, st.session_state.points, reduction_scale=scale)
            st.session_state["mask"] = mask
            st.session_state["result"] = result
            st.session_state["output_path"] = os.path.join(os.getcwd(), 'output', current_file_name)
            st.session_state["create_mask"] = False
            st.rerun()
        else:
            st.warning("Select at least 3 points to create a polygon.")
            st.session_state["create_mask"] = False
    
    if "result" in st.session_state and "mask" in st.session_state:
        st.markdown("### 👁️ Mask Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.image(np.rot90(image), caption="Original Image", use_container_width=True)
        with col2:
            st.image(np.rot90(st.session_state["result"]), caption="Segmented Area", use_container_width=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("💾 Save Mask & Continue", type="primary"):
                # Save mask exactly as drawn, no rotation or transformation
                MaskOperations.save_mask(
                    st.session_state["mask"],
                    affine=st.session_state["affine"],
                    nb_of_slices=st.session_state["nb_of_slices"],
                    file_path=st.session_state["output_path"],
                    points=st.session_state["points"],
                    scale=st.session_state["scale"]
                )
                
                # Mark file as completed
                st.session_state["batch_completed_files"]["draw"].append(current_file_name)
                
                # Move to next file
                st.session_state["batch_current_index"] = current_index + 1
                
                # Clear current file session state
                keys_to_clear = ["points", "mask", "result", "create_mask", "affine", "original_image_path", "nb_of_slices", "scale", "output_path", "current_batch_file"]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.success(f"Mask saved for {current_file}!")
                st.rerun()
