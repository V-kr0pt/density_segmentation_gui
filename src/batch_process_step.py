import os
import numpy as np
import streamlit as st
import nibabel as nib
from PIL import Image
import shutil
from utils import ImageOperations, MaskOperations, ThresholdOperations

def batch_process_step():
    st.header("Step 4: Process Files")
    
    # Add consistent CSS styling
    st.markdown("""
    <style>
    .step-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2d3436;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(168, 237, 234, 0.3);
        border-left: 3px solid #00b894;
    }
    .step-container h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.2rem;
        color: #2d3436;
    }
    .step-container p {
        margin: 0;
        font-size: 0.9rem;
        color: #2d3436;
    }
    .progress-section {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        box-shadow: 0 1px 5px rgba(0,0,0,0.05);
    }
    .file-status {
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    .status-success {
        background: linear-gradient(135deg, #00b894 0%, #55efc4 100%);
        color: white;
        border-left: 3px solid #00b894;
    }
    .status-processing {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2d3436;
        border-left: 3px solid #00b894;
    }
    .status-pending {
        background: linear-gradient(135deg, #ddd 0%, #f1f2f6 100%);
        color: #2d3436;
        border-left: 3px solid #74b9ff;
    }
    .success-container {
        background: linear-gradient(135deg, #00b894 0%, #55efc4 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 184, 148, 0.3);
    }
    .compact-section {
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-container">
        <p>Apply thresholds and generate segmentation masks. This step produces the final output files.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Define input folder
    input_folder = os.path.join(os.getcwd(), 'media')
    
    # Check if we have batch data
    if "batch_files" not in st.session_state:
        st.error("No batch files selected. Please go back to file selection.")
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        return
    
    batch_files = st.session_state["batch_files"]
    completed_draw = st.session_state["batch_completed_files"]["draw"]
    completed_threshold = st.session_state["batch_completed_files"]["threshold"]
    all_completed_completed_process = st.session_state["batch_completed_files"]["process"]
    completed_process = [f for f in all_completed_completed_process if f in st.session_state["batch_files_without_extension"]]
    
    # Check if all previous steps are completed
    if len(completed_draw) < len(batch_files):
        st.warning(f"Please complete all drawing steps first. {len(completed_draw)}/{len(batch_files)} completed.")
        if st.button("← Back to Draw Step"):
            st.session_state["current_step"] = "batch_draw"
            st.rerun()
        return
    
    if len(completed_threshold) < len(batch_files):
        st.warning(f"Please complete all threshold steps first. {len(completed_threshold)}/{len(batch_files)} completed.")
        if st.button("← Back to Threshold Step"):
            st.session_state["current_step"] = "batch_threshold"
            st.rerun()
        return
    
    # Check if we have saved thresholds
    if "batch_final_thresholds" not in st.session_state:
        st.error("No thresholds found. Please complete the threshold step first.")
        if st.button("← Back to Threshold Step"):
            st.session_state["current_step"] = "batch_threshold"
            st.rerun()
        return
    
    total_files = len(batch_files)
    
    # Progress overview
    #st.markdown('<div class="progress-section">', unsafe_allow_html=True)
    st.write(f"### Progress: {len(completed_process)}/{total_files} files completed")
    st.progress(len(completed_process) / total_files)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Files status
    st.write("### Files Status")
    
    final_thresholds = st.session_state["batch_final_thresholds"]
    
    files_to_process = []
    for file in batch_files:
        file_name = file.split('.')[0]
        if file_name in completed_draw and file_name in completed_threshold:
            threshold = final_thresholds.get(file_name, "Not set")
            if file_name in completed_process:
                status = "✅ Completed"
                color = "#28a745"
            else:
                status = "⏳ Ready"
                color = "#ffc107"
                files_to_process.append(file)
                
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 5px; margin: 0.25rem 0; border-left: 3px solid {color};">
                <strong>{file}</strong><br>
                Threshold: <code>{threshold:.3f}</code> | Status: <span style="color: {color};">{status}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 5px; margin: 0.25rem 0; border-left: 3px solid #dc3545;">
                <strong>{file}</strong><br>
                <span style="color: #dc3545;">❌ Missing prerequisites</span>
            </div>
            """, unsafe_allow_html=True)
    
    if len(files_to_process) == 0 and len(completed_process) == total_files:
        st.markdown("""
        <div class="success-container">
            <h4>🎉 All files have been processed successfully!</h4>
            <p>All segmentation masks have been generated and saved to the output directory.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show results summary
        #st.write("### Results")
        #for file in batch_files:
            #file_name = file.split('.')[0]
            #output_path = os.path.join(os.getcwd(), 'output', file_name, 'dense_mask')
            #nifti_path = os.path.join(output_path, 'binary_mask.nii')
            #if os.path.exists(nifti_path):
               # st.markdown(f"""
                #<div style="background: #d4edda; padding: 0.5rem; border-radius: 5px; margin: 0.25rem 0; border-left: 3px solid #28a745;">
                    #<strong>✅ {file}</strong> - Output saved successfully
                #</div>
                #""", unsafe_allow_html=True)
            #else:
                #st.markdown(f"""
                #<div style="background: #f8d7da; padding: 0.5rem; border-radius: 5px; margin: 0.25rem 0; border-left: 3px solid #dc3545;">
                    #<strong>❌ {file}</strong> - Output file not found
                #</div>
                #""", unsafe_allow_html=True)
        
        if st.button("🔄 Start New Batch"):
            # Clear all batch-related session state
            keys_to_remove = [key for key in st.session_state.keys() if key.startswith("batch_")]
            for key in keys_to_remove:
                del st.session_state[key]
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        
        return
    
    if len(files_to_process) == 0:
        st.info("No files pending for processing.")
        if st.button("← Back to Threshold Step"):
            st.session_state["current_step"] = "batch_threshold"
            st.rerun()
        return
    
    # Process all remaining files button
    st.write(f"### Ready to process {len(files_to_process)} files")
    
    if st.button("🚀 Process All Files", type="primary"):
        
        # Create overall progress bar
        overall_progress = st.progress(0, text="Starting batch processing...")
        overall_status = st.empty()
        
        total_operations = len(files_to_process)
        
        for file_idx, file in enumerate(files_to_process):
            file_name = file.split('.')[0] if file.endswith('.nii') or file.endswith('.nii.gz') else file     
            overall_status.text(f"Processing file {file_idx + 1}/{total_operations}: {file}")
            
            try:
                # Get file paths
                original_image_path = os.path.join(input_folder, file)
                output_path = os.path.join(os.getcwd(), 'output', file_name)
                mask_path = os.path.join(output_path, 'dense.nii')
                save_dir = os.path.join(output_path, 'dense_mask')
                
                # Get threshold for this file
                T = final_thresholds[file_name]
                
                # Check if required files exist
                if not os.path.exists(original_image_path):
                    st.error(f"Original image not found: {original_image_path}")
                    continue
                
                if not os.path.exists(mask_path):
                    st.error(f"Mask not found: {mask_path}")
                    continue
                
                # Load data
                _, original_affine, num_slices = ImageOperations.load_image(original_image_path)
                middle_slice_index = num_slices // 2
                middle_image_slice = ImageOperations.load_any_slice(original_image_path, middle_slice_index)                    
                middle_mask_slice = ImageOperations.load_nii_slice(mask_path, middle_slice_index)
                thresholded_img = ThresholdOperations.threshold_image(middle_image_slice, middle_mask_slice, T)
                target_area = MaskOperations.measure_mask_area(thresholded_img)
            
                # Clear existing output directory
                if os.path.exists(save_dir):
                    for f in os.listdir(save_dir):
                        os.remove(os.path.join(save_dir, f))
                
                os.makedirs(save_dir, exist_ok=True)
                
                # Process each slice
                for slice_index in range(num_slices):
                    image_slice = ImageOperations.load_any_slice(original_image_path, slice_index)
                    mask_slice = ImageOperations.load_nii_slice(mask_path, slice_index)
                    mask_slice = np.flip(mask_slice, axis=1)  # Flip the mask slice 
                    adjusted_threshold, thresholded_image = ThresholdOperations.adjust_threshold(image_slice, mask_slice, target_area, slice_index)
                    binary_image = np.where(thresholded_image > 0, 255, 0).astype(np.uint8)
                    filename = f'slice_{slice_index}_threshold_{adjusted_threshold:.4f}.png'
                    filepath = os.path.join(save_dir, filename)
                    Image.fromarray(binary_image.T, mode='L').save(filepath)
                
                # Create NIfTI file
                nifti_path = MaskOperations.create_mask_nifti(save_dir, original_affine)
                
                # Mark as completed
                st.session_state["batch_completed_files"]["process"].append(file_name)
                    
            except Exception as e:
                st.error(f"Error processing {file}: {str(e)}")
                continue
            
            # Update overall progress
            overall_progress.progress((file_idx + 1) / total_operations, 
                                    text=f"Completed {file_idx + 1}/{total_operations} files")
        
        overall_status.text("🎉 Batch processing completed!")
        st.success("All files have been processed successfully!")
        st.rerun()
    
    # Individual file processing section
    st.divider()
    st.markdown("""
    <div class="step-container">
        <h4>🔧 Individual Processing</h4>
        <p>Process files one by one if you prefer more control over each file.</p>
    </div>
    """, unsafe_allow_html=True)
    
    for file in files_to_process:
        file_name = file.split('.')[0] if file.endswith('.nii') or file.endswith('.nii.gz') else file            
        threshold = final_thresholds[file_name]
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 5px; margin: 0.25rem 0; border-left: 3px solid #007bff;">
                <strong>{file}</strong><br>
                Threshold: <code>{threshold:.3f}</code>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("Process", key=f"process_{file_name}"):
                # Process single file
                original_image_path = os.path.join(input_folder, file)
                output_path = os.path.join(os.getcwd(), 'output', file_name)
                mask_path = os.path.join(output_path, 'dense.nii')
                save_dir = os.path.join(output_path, 'dense_mask')
                T = threshold
                
                try:
                    with st.spinner(f"Processing {file}..."):
                        # Processing logic
                        _, original_affine, num_slices = ImageOperations.load_image(original_image_path)
                        middle_slice_index = num_slices // 2
                        middle_image_slice = ImageOperations.load_any_slice(original_image_path, middle_slice_index)
                        middle_mask_slice = ImageOperations.load_nii_slice(mask_path, middle_slice_index)
                        thresholded_img = ThresholdOperations.threshold_image(middle_image_slice, middle_mask_slice, T)
                        target_area = MaskOperations.measure_mask_area(thresholded_img)

                        if os.path.exists(save_dir):
                            for f in os.listdir(save_dir):
                                os.remove(os.path.join(save_dir, f))
                        os.makedirs(save_dir, exist_ok=True)

                        for slice_index in range(num_slices):
                            image_slice = ImageOperations.load_any_slice(original_image_path, slice_index)
                            mask_slice = ImageOperations.load_nii_slice(mask_path, slice_index)
                            mask_slice = np.flip(mask_slice, axis=1) 
                            adjusted_threshold, thresholded_image = ThresholdOperations.adjust_threshold(image_slice, mask_slice, target_area, slice_index)
                            binary_image = np.where(thresholded_image > 0, 255, 0).astype(np.uint8)
                            filename = f'slice_{slice_index}_threshold_{adjusted_threshold:.4f}.png'
                            filepath = os.path.join(save_dir, filename)
                            Image.fromarray(binary_image.T, mode='L').save(filepath)

                        nifti_path = MaskOperations.create_mask_nifti(save_dir, original_affine)
                        st.session_state["batch_completed_files"]["process"].append(file_name)
                        st.success(f"✅ {file} processed successfully!")
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing {file}: {str(e)}")
    
    # Back button
    if st.button("← Back to Threshold Step"):
        st.session_state["current_step"] = "batch_threshold"
        st.rerun()
