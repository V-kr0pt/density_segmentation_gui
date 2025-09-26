
# =========================
# Imports
# =========================
import os
import streamlit as st
from PIL import Image
from new_utils import ImageProcessor, MaskManager
from ImageLoader import UnifiedImageLoader
from streamlit_drawable_canvas import st_canvas
import numpy as np

def batch_draw_step():
    """
    Main function for Step 2: Draw Masks in batch mode.
    Handles navigation, drawing, mask creation, and saving for each file in the batch.
    """

    # =========================
    # UI Header & Styling
    # =========================
    st.header("🎨 Step 2: Draw Masks")
    with open("static/batch_draw_step.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # =========================
    # Batch Data Validation
    # =========================
    if "batch_files" not in st.session_state:
        st.error("No batch files selected. Please go back to file selection.")
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        return

    # =========================
    # Batch State Setup
    # =========================
    batch_files = st.session_state["batch_files"]
    current_index = st.session_state.get("batch_current_index", 0)
    all_completed_draw = st.session_state["batch_completed_files"]["draw"]  # All files with a polygon
    completed_draw = [f for f in all_completed_draw if f in st.session_state["batch_files_without_extension"]]  # Only batch files with a polygon

    # =========================
    # Progress Bar
    # =========================
    total_files = len(batch_files)
    st.write(f"### Progress: {len(completed_draw)}/{total_files} files completed")
    st.progress(len(completed_draw) / total_files)
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # End of Batch Handling
    # =========================
    if current_index >= len(batch_files):
        st.success("🎉 All masks have been drawn!")
        if st.button("→ Continue to Threshold Step"):
            st.session_state["batch_current_index"] = 0
            st.session_state["current_step"] = "batch_threshold"
            st.rerun()
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        return

    # =========================
    # File Navigation & Skipping Completed
    # =========================
    current_file = batch_files[current_index]
    current_file_name = current_file.split('.')[0]
    if current_file_name in completed_draw:
        st.session_state["batch_current_index"] = current_index + 1
        st.rerun()
        return

    # =========================
    # Current File Display
    # =========================
    st.markdown(f"""
    <div class="current-file">
        <h4>📁 {current_file} ({current_index + 1}/{total_files})</h4>
        <p>💡 Navigate between unsaved files • Saved files are locked</p>
    </div>
    """, unsafe_allow_html=True)
    # --- Navigation Buttons ---
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

    # --- Completed Files List ---
    if len(completed_draw) > 0:
        with st.expander(f"✅ Completed Files ({len(completed_draw)})", expanded=False):
            st.caption("*These files are saved and locked*")
            cols = st.columns(3)
            for i, file in enumerate(completed_draw):
                with cols[i % 3]:
                    st.markdown(f"🔒 **{file}.nii**")

    st.divider()

    # =========================
    # File Loading & Preparation
    # =========================
    input_folder = os.path.join(os.getcwd(), 'media')
    file_path = os.path.join(input_folder, current_file)
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return

    # Load image and metadata
    #image, affine, nb_of_slices = ImageOperations.load_image(file_path)
    volume, affine, original_shape = UnifiedImageLoader.load_image(file_path)
    internal_volume_shape = volume.shape 
    nb_of_slices= internal_volume_shape[0]
    image = volume[internal_volume_shape[0]//2, :, :] # loading the central slice as image
    print(f"original shape:{original_shape}")
    print(f"internal_volume_shape:{internal_volume_shape}")
    print(f"image shape: {image.shape}")
    del volume # removing volume to not consume RAM

    # Initialize session state for new file
    if "affine" not in st.session_state or st.session_state.get("current_batch_file") != current_file:
        st.session_state["affine"] = affine
        st.session_state["original_image_path"] = file_path
        st.session_state["nb_of_slices"] = nb_of_slices
        st.session_state["current_batch_file"] = current_file
        st.session_state["points"] = []  # Reset points for new file
        st.session_state["polygons"] = []
        # Clear other file-specific data
        for key in ["mask", "result", "create_mask"]:
            if key in st.session_state:
                del st.session_state[key]

    # =========================
    # Image Scaling & Canvas Setup
    # =========================
    max_width, max_height = 1200, 800
    orig_height, orig_width = image.shape[0], image.shape[1]
    scale = min(max_width / orig_width, max_height / orig_height, 1)
    if "scale" not in st.session_state:
        st.session_state["scale"] = scale

    # Pad image to avoid edge issues 
    padding = 50
    padded_image = np.pad(image,
                           pad_width=((padding, padding), (padding, padding)),
                            mode="constant", constant_values=0)
    padded_image = ImageProcessor.normalize_image(padded_image) 
    
    pil_height = int(padded_image.shape[0] * scale)
    pil_width = int(padded_image.shape[1] * scale)    
    pil_image = Image.fromarray(padded_image).resize((pil_width, pil_height)).rotate(90, expand=True)
    
    if "points" not in st.session_state:
        st.session_state.points = []

    # --- Drawing Canvas ---
    st.markdown("### 🖌️ Drawing Canvas")
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
        key=f"canvas_{current_file}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Canvas Control Buttons ---
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

    # =========================
    # Polygon Extraction from Canvas
    # =========================
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        polygons = []
        for obj in objects:
            if obj["type"] == "path":  # ensure it's a polygon
                poly = obj["path"]
                points_rotated = [(int(p[1]), int(p[2])) for p in poly if len(p) == 3]
                points = []
                for x, y in points_rotated:
                    # removing rotation
                    px, py = rotated_height - y, x

                    # padding removing
                    px -= int(padding*scale)
                    py -= int(padding*scale)

                    # scaling back to original image size
                    px = min(max(px, 0), image.shape[0] - 1)
                    py = min(max(py, 0), image.shape[1] - 1)

                    points.append((px, py))
                polygons.append(points)

        # Save all polygons to session state
        st.session_state.polygons = polygons

    # =========================
    # Mask Creation Logic
    # =========================
    if st.session_state.get("create_mask", False):
        if "polygons" in st.session_state and len(st.session_state.polygons) > 0:
            result, combined_mask = MaskManager.create_combined_mask(image, st.session_state.polygons, scale)
            st.session_state["mask"] = combined_mask
            st.session_state["result"] = result
            st.session_state["output_path"] = os.path.join(os.getcwd(), 'output', current_file_name)
            st.session_state["create_mask"] = False
            st.rerun()
        else:
            st.warning("Select at least 3 points to create a polygon.")
            st.session_state["create_mask"] = False

    # =========================
    # Mask Preview & Save
    # =========================
    if "result" in st.session_state and "mask" in st.session_state:
        st.markdown("### 👁️ Mask Preview")
        col1, col2 = st.columns(2)

        present_img = ImageProcessor.normalize_image(np.rot90(image))
        present_result = ImageProcessor.normalize_image(np.rot90(st.session_state["result"]))

        with col1:
            st.image(present_img, caption="Original Image", use_container_width=True)
        with col2:
            st.image(present_result, caption="Segmented Area", use_container_width=True)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("💾 Save Mask & Continue", type="primary"):
                # Save mask exactly as drawn, no rotation or transformation
                _ = MaskManager.save_mask(
                        mask=st.session_state["mask"],
                        original_shape=original_shape,
                        nb_of_slices=st.session_state["nb_of_slices"],
                        affine=st.session_state["affine"],
                        file_path=st.session_state["output_path"],
                        #points=st.session_state["points"], in old version we save a .json file with the points and scale info
                        #scale=st.session_state["scale"]
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
