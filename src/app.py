import os
import streamlit as st
from PIL import Image 
from streamlit_drawable_canvas import st_canvas
from imageloader import ImageLoader

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
    
    st.write("Draw a polygon on the image to segment:")


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







if __name__ == "__main__":
    main()
