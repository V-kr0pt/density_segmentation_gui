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
        middle_image_slice = np.rot90(ImageOperations.load_nii_slice(original_image_path, middle_slice_index))
        middle_mask_slice = ImageOperations.load_nii_slice(mask_path, middle_slice_index)
        target_area = MaskOperations.measure_mask_area(ThresholdOperations.threshold_image(middle_image_slice, middle_mask_slice, T))
        
        original_affine = nib.load(original_image_path).affine
        
        for slice_index in range(num_slices):
            status_text.text(f"Processing slice {slice_index+1}/{num_slices}")
            progress_bar.progress((slice_index+1)/num_slices)
            
            image_slice = np.rot90(ImageOperations.load_nii_slice(original_image_path, slice_index))
            mask_slice = ImageOperations.load_nii_slice(mask_path, slice_index)
            adjusted_threshold, thresholded_image = ThresholdOperations.adjust_threshold(image_slice, mask_slice, target_area, slice_index)
            binary_image = np.where(thresholded_image > 0, 255, 0).astype(np.uint8)
            filename = f'slice_{slice_index}_threshold_{adjusted_threshold:.4f}.png'
            filepath = os.path.join(save_dir, filename)
            Image.fromarray(binary_image, mode='L').save(filepath)
        
        nifti_path = MaskOperations.create_mask_nifti(save_dir, original_affine)
        st.success(f"Processing completed! NIfTI file saved at: {nifti_path}")
    
    if st.button("↩️ Back to Threshold Adjustment"):
        st.session_state["current_step"] = "threshold"
        st.rerun()