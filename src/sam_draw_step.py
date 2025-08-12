"""
SAM2 Draw Step - Individual file drawing (similar to batch_draw_step but for single file)
"""
import os
import streamlit as st
from PIL import Image
from utils import ImageOperations, MaskOperations
from streamlit_drawable_canvas import st_canvas
import numpy as np

def sam_draw_step():
    st.header("SAM2 Processing - Step 1: Draw Mask")
    
    # Check if we have SAM file selected
    if "sam_selected_file" not in st.session_state:
        st.error("No file selected. Please go back to file selection.")
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "sam"
            st.rerun()
        return
    
    selected_file = st.session_state["sam_selected_file"]
    file_path = st.session_state["sam_file_path"]
    file_name = st.session_state["sam_file_name"]
    
    st.write(f"### Currently processing: `{selected_file}`")
    
    # Check if already completed
    output_path = os.path.join(os.getcwd(), 'output', file_name)
    mask_path = os.path.join(output_path, 'dense.nii')
    
    if os.path.exists(mask_path):
        st.info("✅ Mask already exists for this file.")
        if st.button("🔄 Redraw Mask"):
            # Clear existing mask
            if os.path.exists(mask_path):
                os.remove(mask_path)
            # Clear session state
            keys_to_clear = ["points", "mask", "result", "create_mask", "affine", "original_image_path", "nb_of_slices", "scale", "output_path"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        if st.button("➡️ Continue to Threshold Step"):
            st.session_state["current_step"] = "sam_threshold"
            st.rerun()
        
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "sam"
            st.rerun()
        return
    
    # Clean section button
    if st.button("🧹 Clean Current Data"):
        keys_to_clear = ["points", "mask", "result", "create_mask", "affine", "original_image_path", "nb_of_slices", "scale", "output_path"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Current data cleared.")
        st.rerun()
    
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return
    
    # Load image (same logic as batch_draw_step)
    image, affine, nb_of_slices = ImageOperations.load_image(file_path)
    
    if "affine" not in st.session_state or st.session_state.get("current_sam_file") != selected_file:
        st.session_state["affine"] = affine
        st.session_state["original_image_path"] = file_path
        st.session_state["nb_of_slices"] = nb_of_slices
        st.session_state["current_sam_file"] = selected_file
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
    
    st.write("Draw a polygon on the image to segment:")
    
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
        key="sam_canvas"
    )
    
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if len(objects) > 0:
            polygon = objects[-1]["path"]
            # Transform points from rotated canvas back to original orientation
            points_rotated = [(int(p[1]), int(p[2])) for p in polygon if len(p) == 3]
            # For 90 degree rotation (counterclockwise), new_x = y, new_y = width - x
            points = [(rotated_height - y, x) for x, y in points_rotated]
            st.session_state.points = points
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Create mask on the selected area"):
            st.session_state["create_mask"] = True
    
    if st.session_state.get("create_mask", False):
        if len(st.session_state.points) >= 3:
            result, mask = MaskOperations.create_mask(image, st.session_state.points, reduction_scale=scale)
            st.session_state["mask"] = mask
            st.session_state["result"] = result
            st.session_state["output_path"] = os.path.join(os.getcwd(), 'output', file_name)
            st.session_state["create_mask"] = False
            st.rerun()
        else:
            st.warning("Select at least 3 points.")
    
    if "result" in st.session_state and "mask" in st.session_state:
        st.subheader("Mask Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.image(np.rot90(image), caption="Original Image", use_container_width=True)
        with col2:
            st.image(np.rot90(st.session_state["result"]), caption="Segmented Area", use_container_width=True)
        
        if st.button("✅ Save Mask and Continue to Threshold Step"):
            # Save mask exactly as drawn, no rotation or transformation
            MaskOperations.save_mask(
                st.session_state["mask"],
                affine=st.session_state["affine"],
                nb_of_slices=st.session_state["nb_of_slices"],
                file_path=st.session_state["output_path"],
                points=st.session_state["points"],
                scale=st.session_state["scale"]
            )
            
            # Clear current file session state
            keys_to_clear = ["points", "mask", "result", "create_mask", "affine", "original_image_path", "nb_of_slices", "scale", "output_path", "current_sam_file"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.success(f"Mask saved for {selected_file}!")
            
            # Move to threshold step
            st.session_state["current_step"] = "sam_threshold"
            st.rerun()
    
    # Back button
    if st.button("← Back to File Selection"):
        st.session_state["current_step"] = "sam"
        st.rerun()
