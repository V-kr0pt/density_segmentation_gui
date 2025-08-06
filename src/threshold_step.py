#from tqdm import tqdm
import os
import io
import streamlit as st
from utils import ImageOperations, ThresholdOperations

def threshold_step():
    st.header("Step 2: Adjust Threshold")
    
    if "output_path" not in st.session_state:
        st.error("No mask found. Please go back to Step 1.")
        return
    
    mask_path = os.path.join(st.session_state["output_path"], 'dense.nii')
    original_image_path = st.session_state["original_image_path"]
    
    try:
        img = ImageOperations.load_nii_central_slice(original_image_path)
        msk = ImageOperations.load_nii_central_slice(mask_path, flip=True)
    except Exception as e:
        st.error(f"Error loading images: {str(e)}")
        return
    
    width_options = [400, 500, 600, 700, 800, 900, 1000]
    selected_width = st.selectbox("Select image width", width_options, index=2)

    
    threshold = st.slider(
        "Select threshold value",
        min_value=0.0,
        max_value=1.0,
        value=0.38,
        step=0.01,
        key="threshold_slider"
    )

    fig = ThresholdOperations.display_thresholded_slice(img, msk, threshold)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    cols = st.columns([1, 1, 1])
    with cols[1]:
        st.image(buf, caption="Thresholded Central Slice", width=selected_width)
    
    if st.button("💾 Save Thresholded Mask"):
        save_dir = os.path.join(st.session_state["output_path"], 'dense_mask')
        os.makedirs(save_dir, exist_ok=True)
        
        
        # Agora podemos salvar o valor sem conflito
        st.session_state["final_threshold"] = threshold
        st.session_state["current_step"] = "process"
        st.success(f"Threshold {threshold:.2f} saved. Proceeding to processing.")
        st.rerun()
    
    if st.button("↩️ Back to Mask Drawing"):
        st.session_state["current_step"] = "draw"
        st.rerun()
    
    if st.button("🏠 Back to Main Menu"):
        st.session_state.clear()
        st.session_state["current_step"] = "mode_selection"
        st.rerun()