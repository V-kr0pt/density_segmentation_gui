import os
import numpy as np
import streamlit as st
import nibabel as nib
from PIL import Image
from utils import ImageOperations, MaskOperations, ThresholdOperations

def process_step():
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

        # Carrega o slice central da imagem e máscara
        middle_image_slice = ImageOperations.load_nii_slice(original_image_path, middle_slice_index)
        middle_mask_slice = ImageOperations.load_nii_slice(mask_path, middle_slice_index)

        # Ajusta orientação para o slice central se necessário
        if middle_image_slice.shape != middle_mask_slice.shape:
            if middle_image_slice.shape == middle_mask_slice.T.shape:
                middle_mask_slice = middle_mask_slice.T
            elif middle_image_slice.T.shape == middle_mask_slice.shape:
                middle_image_slice = middle_image_slice.T
            else:
                st.error(f"Shape mismatch irreconciliável no slice central: imagem {middle_image_slice.shape} vs máscara {middle_mask_slice.shape}")
                return
        
        target_area = MaskOperations.measure_mask_area(
            ThresholdOperations.threshold_image(middle_image_slice, middle_mask_slice, T)
        )
        
        original_affine = nib.load(original_image_path).affine
        os.makedirs(save_dir, exist_ok=True)

        for slice_index in range(num_slices):
            status_text.text(f"Processing slice {slice_index+1}/{num_slices}")
            progress_bar.progress((slice_index+1)/num_slices)
            
            image_slice = ImageOperations.load_nii_slice(original_image_path, slice_index)
            mask_slice = ImageOperations.load_nii_slice(mask_path, slice_index)

            # Ajusta orientação para o slice atual se necessário
            if image_slice.shape != mask_slice.shape:
                if image_slice.shape == mask_slice.T.shape:
                    mask_slice = mask_slice.T
                elif image_slice.T.shape == mask_slice.shape:
                    image_slice = image_slice.T
                else:
                    st.error(f"Shape mismatch irreconciliável na slice {slice_index}: imagem {image_slice.shape} vs máscara {mask_slice.shape}")
                    return

            st.write(f"Slice {slice_index} shapes - image: {image_slice.shape}, mask: {mask_slice.shape}")
            st.write(f"Mask pixels > 0: {np.count_nonzero(mask_slice)}")
            
            adjusted_threshold, thresholded_image = ThresholdOperations.adjust_threshold(
                image_slice, mask_slice, target_area, slice_index
            )
            
            active_pixels = np.count_nonzero(thresholded_image)
            st.write(f"Slice {slice_index} pixels ativados após threshold: {active_pixels}")
            
            # Transposição só da imagem, sem alterar a máscara
            transposed_thresholded_image = thresholded_image.T
            
            binary_image = np.where(transposed_thresholded_image > 0, 255, 0).astype(np.uint8)
            filename = f'slice_{slice_index}_threshold_{adjusted_threshold:.4f}.png'
            filepath = os.path.join(save_dir, filename)
            Image.fromarray(binary_image, mode='L').save(filepath)
        
        nifti_path = MaskOperations.create_mask_nifti(save_dir, original_affine)
        st.success(f"Processing completed! NIfTI file salvo em: {nifti_path}")
        st.balloons()
    
    if st.button("↩️ Back to Threshold Adjustment"):
        st.session_state["current_step"] = "threshold"
        st.rerun()
