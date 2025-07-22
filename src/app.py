import os
import streamlit as st
from PIL import Image 
from streamlit_drawable_canvas import st_canvas
from utils import ImageLoader, MaskOperations

def file_selector(folder_path=os.path.join(os.getcwd(), 'media')):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select a file', filenames)
        return os.path.join(folder_path, selected_filename)
        
def main():
    st.set_page_config(layout="wide")
    st.title("# Density Segmentation GUI")

    # Select file and load image
    file_path = file_selector()
    image = ImageLoader.load_image(file_path)

    # Define the maximum width and height for the image display
    max_width, max_height = 1200, 800
    orig_height, orig_width = image.shape[0], image.shape[1]

    # Calculate the scale to fit the image within the max dimensions
    scale = min(max_width / orig_width, max_height / orig_height, 1)
    pil_width = int(orig_width * scale)
    pil_height = int(orig_height * scale)

    # Resize the image for display
    pil_image = Image.fromarray(image).resize((pil_width, pil_height))

    if "points" not in st.session_state:
        st.session_state.points = []
    
        # Create canvas for drawing allowing to clear the section
    if "clear_canvas" not in st.session_state:
        st.session_state.clear_canvas = False

    # Clear the canvas if requested
    if st.button("Clean Section"):
        st.session_state.points = []
        st.session_state.clear_canvas = True
        st.success("Selection cleared.")

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

    # Reset the clear flag after rendering the canvas
    if st.session_state.clear_canvas:
        st.session_state.clear_canvas = False

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if len(objects) > 0:
            polygon = objects[-1]["path"]
            points = [(int(p[1]), int(p[2])) for p in polygon if len(p) == 3]
            st.session_state.points = points

    if st.button("Create mask on the selected area"):
        if len(st.session_state.points) >= 3:
            # Create a mask from the drawn polygon
            result, mask = MaskOperations.create_mask(image, st.session_state.points, reduction_scale=scale)

            # Display the original image and the segmented area
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col2:
                st.image(result, caption="Segmented Area", use_container_width=True)

            # Save the mask
            if st.button("Save Mask"):
                MaskOperations.save_mask(mask, file_path=os.path.join(os.getcwd(), 'output', 'mask.png'))
                st.success("Mask saved successfully.")

        else:
            st.warning("Select at least 3 points.")



if __name__ == "__main__":
    main()
