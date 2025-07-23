import os
import streamlit as st
from PIL import Image 
from streamlit_drawable_canvas import st_canvas
from utils import ImageLoader, MaskOperations

def file_selector(folder_path=os.path.join(os.getcwd(), 'media')):
        filenames = [f for f in os.listdir(folder_path) if f.endswith('.nii')]
        selected_filename = st.selectbox('Select a file', filenames)
        return selected_filename, os.path.join(folder_path, selected_filename)

def main():
    st.set_page_config(layout="wide")
    st.title("# Density Segmentation GUI")

    # Clear all if requested
    if st.button("Clean Section"):
        st.session_state.clear()
        st.success("Selection cleared.")

    # Select file and load image
    selected_filename, file_path = file_selector()
    image, affine = ImageLoader.load_image(file_path)

    if "affine" not in st.session_state:
        st.session_state["affine"] = affine

    # Define the maximum width and height for the image display
    max_width, max_height = 1200, 800
    orig_height, orig_width = image.shape[0], image.shape[1]

    # Calculate the scale to fit the image within the max dimensions
    scale = min(max_width / orig_width, max_height / orig_height, 1)

    if "scale" not in st.session_state:
        st.session_state["scale"] = scale
    
    pil_width = int(orig_width * scale)
    pil_height = int(orig_height * scale)

    # Resize the image for display
    pil_image = Image.fromarray(image).resize((pil_width, pil_height))

    if "points" not in st.session_state:
        st.session_state.points = []
    
    st.write("Draw a polygon on the image to segment:")
    
    # Render the canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="red",
        background_image=pil_image,
        update_streamlit=True,
        height=pil_height,
        width=pil_width,
        drawing_mode="polygon",
        key="canvas"
    )

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if len(objects) > 0:
            polygon = objects[-1]["path"]
            points = [(int(p[1]), int(p[2])) for p in polygon if len(p) == 3]
            st.session_state.points = points

    if st.button("Create mask on the selected area"):
        st.session_state["create_mask"] = True

    if st.session_state.get("create_mask", False):
        if len(st.session_state.points) >= 3:
            # Create a mask from the drawn polygon
            result, mask = MaskOperations.create_mask(image, st.session_state.points, reduction_scale=scale)
            st.session_state["mask"] = mask
            st.session_state["result"] = result
            st.session_state["output_path"] = os.path.join(os.getcwd(), 'output', selected_filename.split('.')[0])

            # Display the original image and the segmented area
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col2:
                st.image(result, caption="Segmented Area", use_container_width=True)

            st.session_state["create_mask"] = False
            
        else:
            st.warning("Select at least 3 points.")

    # Save the mask if requested
    if st.button("Save Mask"):
        st.session_state["save_mask"] = True

    if st.session_state.get("save_mask", False):
        st.session_state["save_mask"] = False
        if "mask" in st.session_state:
            st.write(f"Saving mask to: {st.session_state.output_path}")

            # We have to create the local variables to avoid issues with Streamlit's session state
            MaskOperations.save_mask(st.session_state.mask, affine=st.session_state.affine, 
                                        file_path=st.session_state.output_path,
                                        points=st.session_state.points, 
                                        scale=st.session_state.scale)
            st.success("Mask saved successfully.")
            st.session_state.points = []
            st.session_state.mask = None
            st.session_state.result = None            
        else:
            st.error("No mask to save. Please create a mask first.")
        

if __name__ == "__main__":
    main()
