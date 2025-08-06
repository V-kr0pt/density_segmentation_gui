import os
import streamlit as st
from PIL import Image
from utils import ImageOperations, MaskOperations
from streamlit_drawable_canvas import st_canvas
import numpy as np

def file_selector(folder_path=os.path.join(os.getcwd(), 'media'), only_not_done=True):
        already_done_files = os.listdir(os.path.join(os.getcwd(), 'output'))
        filenames = [f for f in os.listdir(folder_path) if f.endswith('.nii')]

        if only_not_done:
            filenames = [f for f in filenames if f.split('.')[0] not in already_done_files]

        selected_filename = st.selectbox('Select a file', filenames)

        if len(filenames) == 0:
            st.warning("No files available to select.")
            return None, None
        else:
            return selected_filename, os.path.join(folder_path, selected_filename)


def draw_step():
    st.header("Step 1: Draw Mask")
    
    if st.button("Clean Section"):
        st.session_state.clear()
        st.success("Selection cleared.")
        st.rerun()
    
    if st.button("🏠 Back to Main Menu"):
        st.session_state.clear()
        st.session_state["current_step"] = "mode_selection"
        st.rerun()

    input_folder = os.path.join(os.getcwd(), 'media')
    only_not_done = st.checkbox("Show only not done files")
    selected_filename, file_path = file_selector(folder_path=input_folder, only_not_done=only_not_done)
    if selected_filename is None:
        st.warning(f"Please add a new .nii file in {input_folder} to continue.")
        return False

    image, affine, nb_of_slices = ImageOperations.load_image(file_path)

    if "affine" not in st.session_state:
        st.session_state["affine"] = affine
        st.session_state["original_image_path"] = file_path
        st.session_state["nb_of_slices"] = nb_of_slices

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
        key="canvas"
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
            st.session_state["output_path"] = os.path.join(os.getcwd(), 'output', selected_filename.split('.')[0])
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
        
        if st.button("✅ Save Mask and Proceed to Thresholding"):
            # Save mask exactly as drawn, no rotation or transformation
            MaskOperations.save_mask(
                st.session_state["mask"],
                affine=st.session_state["affine"],
                nb_of_slices=st.session_state["nb_of_slices"],
                file_path=st.session_state["output_path"],
                points=st.session_state["points"],
                scale=st.session_state["scale"]
            )
            st.session_state["current_step"] = "threshold"
            st.rerun()
    
    return False