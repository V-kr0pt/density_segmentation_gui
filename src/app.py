import os
import numpy as np
import streamlit as st
from PIL import Image 
import nibabel as nib
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from utils import ImageLoader, MaskOperations
import gc
from tqdm import tqdm

# ================== Helper Functions ==================

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

def rearrange_dimensions(nii_data):
    if nii_data.shape[0] > nii_data.shape[-1]:
        nii_data = np.transpose(nii_data, (2, 0, 1))
    return nii_data

def load_nii_central_slice(file_path, dtype=np.float32):
    nii = nib.load(file_path).get_fdata()
    nii = rearrange_dimensions(nii)
    idx = nii.shape[0] // 2
    return nii[idx].astype(dtype)

def normalize_data(data):
    mn, mx = data.min(), data.max()
    return (data - mn) / (mx - mn)

def threshold_image(img_slice, mask_slice, threshold):
    norm = normalize_data(img_slice)
    return (norm > threshold) & (mask_slice > 0)

def display_thresholded_slice(img, mask, threshold):
    bin_mask = threshold_image(img, mask, threshold)
    
    rotated_img = np.rot90(img, k=1)
    rotated_mask = np.rot90(bin_mask, k=1)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rotated_img, cmap='gray')
    ax.imshow(rotated_mask, cmap='jet', alpha=0.2)
    ax.text(0.5, 0.05, f'Threshold: {threshold:.2f}',
           ha='center', va='center',
           transform=ax.transAxes,
           color='white', fontsize=16)
    ax.axis('off')
    
    return fig

# ================== Processing Functions ==================

def load_nii_slice(file_path, slice_index, dtype=np.float32):
    nii_obj = nib.load(file_path)
    nii_data = nii_obj.get_fdata()
    slice_data = nii_data[slice_index, :, :]
    del nii_data, nii_obj
    gc.collect()
    return slice_data.astype(dtype)

def measure_mask_area(mask):
    return np.count_nonzero(mask)

def adjust_threshold(image_slice, mask_slice, target_area, slice_index):
    threshold = 0.8
    step = 0.01
    best_threshold = threshold
    best_diff = float('inf')

    while threshold >= 0:
        thresholded_image = threshold_image(image_slice, mask_slice, threshold)
        area = measure_mask_area(thresholded_image)
        diff = abs(area - target_area)

        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold

        threshold -= step

    return best_threshold, threshold_image(image_slice, mask_slice, best_threshold)

def create_mask_nifti(folder_path, original_affine):
    images = [np.array(Image.open(os.path.join(folder_path, f)).convert('L')) 
              for f in sorted(os.listdir(folder_path), 
              key=lambda x: int(x.split('_')[1])) if f.endswith('.png')]
    volume = np.stack(images, axis=0)
    transposed_flipped_volume = np.flip(np.transpose(volume, (0, 2, 1)), axis=2)
    nib.save(nib.Nifti1Image(transposed_flipped_volume, original_affine), 
             os.path.join(folder_path, 'mask.nii'))
    return os.path.join(folder_path, 'mask.nii')

# ================== App Steps ==================

def step_draw_mask():
    st.header("Step 1: Draw Mask")
    
    if st.button("Clean Section"):
        st.session_state.clear()
        st.success("Selection cleared.")
        st.rerun()

    input_folder = os.path.join(os.getcwd(), 'media')
    only_not_done = st.checkbox("Show only not done files")
    selected_filename, file_path = file_selector(folder_path=input_folder, only_not_done=only_not_done)
    if selected_filename is None:
        st.warning(f"Please add a new .nii file in {input_folder} to continue.")
        return False

    image, affine, nb_of_slices = ImageLoader.load_image(file_path)

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

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if len(objects) > 0:
            polygon = objects[-1]["path"]
            points = [(int(p[1]), int(p[2])) for p in polygon if len(p) == 3]
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
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(st.session_state["result"], caption="Segmented Area", use_container_width=True)
        
        if st.button("✅ Save Mask and Proceed to Thresholding"):
            MaskOperations.save_mask(
                st.session_state.mask, 
                affine=st.session_state.affine,
                nb_of_slices=st.session_state["nb_of_slices"], 
                file_path=st.session_state["output_path"],
                points=st.session_state.points, 
                scale=st.session_state.scale
            )
            st.session_state["current_step"] = "threshold"
            st.rerun()
    
    return False

def step_threshold():
    st.header("Step 2: Adjust Threshold")
    
    if "output_path" not in st.session_state:
        st.error("No mask found. Please go back to Step 1.")
        return
    
    mask_path = os.path.join(st.session_state["output_path"], 'dense.nii')
    original_image_path = st.session_state["original_image_path"]
    
    try:
        img = load_nii_central_slice(original_image_path)
        msk = load_nii_central_slice(mask_path)
    except Exception as e:
        st.error(f"Error loading images: {str(e)}")
        return
    
    # Modificação principal: usar uma chave diferente para o slider
    threshold = st.slider(
        "Select threshold value",
        min_value=0.0,
        max_value=1.0,
        value=0.38,
        step=0.01,
        key="threshold_slider"  # Chave diferente da session_state que será usada
    )
    
    fig = display_thresholded_slice(img, msk, threshold)
    st.pyplot(fig)
    
    if st.button("💾 Save Thresholded Mask"):
        save_dir = os.path.join(st.session_state["output_path"], 'dense_mask')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save central slice preview
        preview_path = os.path.join(save_dir, 'preview.png')
        plt.imsave(preview_path, np.rot90(threshold_image(img, msk, threshold), k=1), cmap='gray')
        
        # Agora podemos salvar o valor sem conflito
        st.session_state["final_threshold"] = threshold
        st.session_state["current_step"] = "process"
        st.success(f"Threshold {threshold:.2f} saved. Proceeding to processing.")
        st.rerun()
    
    if st.button("↩️ Back to Mask Drawing"):
        st.session_state["current_step"] = "draw"
        st.rerun()

def step_process():
    st.header("Step 3: Process All Slices")
    
    if "output_path" not in st.session_state or "final_threshold" not in st.session_state:
        st.error("Missing required data. Please go back to Step 1.")
        return
    
    original_image_path = st.session_state["original_image_path"]
    mask_path = os.path.join(st.session_state["output_path"], 'dense.nii')
    save_dir = os.path.join(st.session_state["output_path"], 'dense_mask')
    T = st.session_state["final_threshold"]  
    
    if st.button("Start Processing All Slices"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        num_slices = nib.load(original_image_path).shape[0]
        middle_slice_index = num_slices // 2
        middle_image_slice = load_nii_slice(original_image_path, middle_slice_index)
        middle_mask_slice = load_nii_slice(mask_path, middle_slice_index)
        target_area = measure_mask_area(threshold_image(middle_image_slice, middle_mask_slice, T))
        
        original_affine = nib.load(original_image_path).affine
        
        for slice_index in range(num_slices):
            status_text.text(f"Processing slice {slice_index+1}/{num_slices}")
            progress_bar.progress((slice_index+1)/num_slices)
            
            image_slice = load_nii_slice(original_image_path, slice_index)
            mask_slice = load_nii_slice(mask_path, slice_index)
            adjusted_threshold, thresholded_image = adjust_threshold(image_slice, mask_slice, target_area, slice_index)
            
            rotated_thresholded_image = np.rot90(thresholded_image)
            binary_image = np.where(rotated_thresholded_image > 0, 255, 0).astype(np.uint8)
            filename = f'slice_{slice_index}_threshold_{adjusted_threshold:.4f}.png'
            filepath = os.path.join(save_dir, filename)
            Image.fromarray(binary_image, mode='L').save(filepath)
        
        nifti_path = create_mask_nifti(save_dir, original_affine)
        st.success(f"Processing completed! NIfTI file saved at: {nifti_path}")
        st.balloons()
    
    if st.button("↩️ Back to Threshold Adjustment"):
        st.session_state["current_step"] = "threshold"
        st.rerun()

# ================== Main App ==================

def main():
    st.set_page_config(layout="wide")
    st.title("# Density Segmentation GUI")
    
    if "current_step" not in st.session_state:
        st.session_state["current_step"] = "draw"
    
    if st.session_state["current_step"] == "draw":
        step_draw_mask()
    elif st.session_state["current_step"] == "threshold":
        step_threshold()
    elif st.session_state["current_step"] == "process":
        step_process()

if __name__ == "__main__":
    main()