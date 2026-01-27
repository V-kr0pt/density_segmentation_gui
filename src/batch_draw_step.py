# =========================
# Imports
# =========================
import os
import streamlit as st
from PIL import Image
from new_utils import ImageProcessor, MaskManager, DisplayTransform
from ImageLoader import UnifiedImageLoader
from streamlit_drawable_canvas import st_canvas
import numpy as np

def batch_draw_step():
    """
    Step 2: Draw Masks in batch mode.
    Uses DisplayTransform to properly handle coordinate transformations.
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
    all_completed_draw = st.session_state["batch_completed_files"]["draw"]
    completed_draw = [f for f in all_completed_draw if f in st.session_state["batch_files_without_extension"]]

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
        <h4>📄 {current_file} ({current_index + 1}/{total_files})</h4>
        <p>💡 Navigate between unsaved files • Saved files are locked</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Navigation Buttons ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if current_index > 0 and st.button("← Previous", help="Navigate to previous unsaved file"):
            st.session_state["batch_current_index"] = current_index - 1
            # Clear current file session state
            keys_to_clear = ["display_transform", "mask", "result", "create_mask", 
                           "affine", "original_image_path", "slice_index", "output_path",
                           "original_shape", "native_polygons"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    with col3:
        next_file_index = current_index + 1
        if next_file_index < len(batch_files) and st.button("Next →", help="Navigate to next unsaved file"):
            st.session_state["batch_current_index"] = next_file_index
            # Clear current file session state
            keys_to_clear = ["display_transform", "mask", "result", "create_mask",
                           "affine", "original_image_path", "slice_index", "output_path",
                           "original_shape", "native_polygons"]
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
    input_folder = st.session_state.get("main_input_folder", os.path.join(os.getcwd(), "media"))
    file_path = os.path.join(input_folder, current_file)

    if not os.path.exists(input_folder):
        st.error(f"The selected main input folder does not exist: {input_folder}")
        return
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return

    # Load image in NATIVE orientation
    image_slice, affine, original_shape, slice_index = UnifiedImageLoader.load_slice(file_path)
    
    print(f"=== File: {current_file} ===")
    print(f"Original shape: {original_shape}")
    print(f"Loaded slice shape: {image_slice.shape}")
    print(f"Slice index: {slice_index}")
    print(f"Slice dimension: {np.argmin(original_shape)}")

    # Initialize session state for new file
    if "original_image_path" not in st.session_state or st.session_state.get("current_batch_file") != current_file:
        st.session_state["affine"] = affine
        st.session_state["original_image_path"] = file_path
        st.session_state["original_shape"] = original_shape
        st.session_state["slice_index"] = slice_index
        st.session_state["current_batch_file"] = current_file
        st.session_state["native_polygons"] = []
        # Clear other file-specific data
        for key in ["mask", "result", "create_mask", "display_transform"]:
            if key in st.session_state:
                del st.session_state[key]

    # =========================
    # Display Transformation Setup
    # =========================
    if "display_transform" not in st.session_state:
        st.session_state["display_transform"] = DisplayTransform(padding=50)
    
    transform = st.session_state["display_transform"]
    
    # Prepare image for display
    pil_image = transform.prepare_for_display(image_slice, max_width=1200, max_height=800)
    rotated_width, rotated_height = pil_image.size

    # --- Drawing Canvas ---
    st.markdown("### 🖌️ Drawing Canvas")
    
    # Drawing instructions (collapsible)
    with st.expander("Drawing Instructions", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
    <small>

    **Mouse**
    - Left click: add a point  
    - Double click: undo last segment  
    - Right click: close polygon  

    </small>
    """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
    <small>

    **Mac trackpad**
    - Single tap: add a point  
    - Double tap: undo last segment  
    - Two-finger tap: close polygon  

    </small>
    """, unsafe_allow_html=True)

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="red",
        background_image=pil_image,
        update_streamlit=True,
        height=rotated_height,
        width=rotated_width,
        drawing_mode="polygon",
        display_toolbar=True,
        key=f"canvas_{current_file}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Canvas Control Buttons ---
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎨 Create Mask"):
            st.session_state["create_mask"] = True
    with col2:
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()

    # =========================
    # Polygon Extraction from Canvas
    # =========================
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        
        # If canvas was cleared (no objects), clear all related state
        if len(objects) == 0:
            st.session_state["native_polygons"] = []
            # Clear mask-related state if it exists
            for key in ["mask", "result", "create_mask"]:
                if key in st.session_state:
                    del st.session_state[key]
        else:
            # Extract polygons from canvas and convert to native coordinates
            native_polygons = []
            
            for obj in objects:
                if obj["type"] == "path":
                    poly = obj["path"]
                    # Extract canvas coordinates
                    canvas_points = [(int(p[1]), int(p[2])) for p in poly if len(p) == 3]
                    
                    # Convert to native image coordinates using DisplayTransform
                    native_points = transform.canvas_to_native_coords(
                        canvas_points, rotated_width, rotated_height
                    )
                    
                    if len(native_points) >= 3:
                        native_polygons.append(native_points)
            
            # Save native polygons to session state
            st.session_state["native_polygons"] = native_polygons

    # =========================
    # Mask Creation Logic
    # =========================
    if st.session_state.get("create_mask", False):
        if "native_polygons" in st.session_state and len(st.session_state["native_polygons"]) > 0:
            # Create mask in NATIVE orientation
            result, combined_mask = MaskManager.create_combined_mask(
                image_slice, 
                st.session_state["native_polygons"]
            )
            
            st.session_state["mask"] = combined_mask
            st.session_state["result"] = result
            st.session_state["output_path"] = os.path.join(os.getcwd(), 'output', current_file_name)
            st.session_state["create_mask"] = False
            st.rerun()
        else:
            st.warning("Please draw at least one polygon with 3 or more points.")
            st.session_state["create_mask"] = False

    # =========================
    # Mask Preview & Save
    # =========================
    if "result" in st.session_state and "mask" in st.session_state:
        st.markdown("### 👁️ Mask Preview")
        col1, col2 = st.columns(2)

        # Display with rotation for visualization consistency
        present_img = ImageProcessor.normalize_image(np.rot90(image_slice))
        present_result = ImageProcessor.normalize_image(np.rot90(st.session_state["result"]))

        with col1:
            st.image(present_img, caption="Original Image", use_container_width=True)
        with col2:
            st.image(present_result, caption="Segmented Area", use_container_width=True)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("💾 Save Mask & Continue", type="primary"):
                # Save mask in NATIVE orientation with proper dimension handling
                _ = MaskManager.save_mask(
                    mask_2d=st.session_state["mask"],
                    original_shape=st.session_state["original_shape"],
                    affine=st.session_state["affine"],
                    file_path=st.session_state["output_path"]
                )
                
                # Mark file as completed
                st.session_state["batch_completed_files"]["draw"].append(current_file_name)
                
                # Move to next file
                st.session_state["batch_current_index"] = current_index + 1
                
                # Clear current file session state
                keys_to_clear = ["display_transform", "mask", "result", "create_mask",
                               "affine", "original_image_path", "slice_index", "output_path",
                               "original_shape", "current_batch_file", "native_polygons"]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.success(f"Mask saved for {current_file}!")
                st.rerun()