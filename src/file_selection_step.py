import os
import streamlit as st
import json

def file_selection_step():
    st.header("📁 Step 1: Select Files")
    
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
    .file-item {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2d3436;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        border-left: 3px solid #00b894;
        font-size: 0.9rem;
        box-shadow: 0 2px 8px rgba(168, 237, 234, 0.3);
    }
    .file-status {
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    .status-complete {
        background: linear-gradient(135deg, #00b894 0%, #55efc4 100%);
        color: white;
    }
    .status-partial {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        color: white;
    }
    .status-pending {
        background: linear-gradient(135deg, #ddd 0%, #f1f2f6 100%);
        color: #2d3436;
    }
    .compact-section {
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-container">
        <p>Choose NIfTI (.nii or .nii.gz) files or DICOM folders for batch processing.</p>
    </div>
    """, unsafe_allow_html=True)
    
    input_folder = os.path.join(os.getcwd(), 'media')
    output_folder = os.path.join(os.getcwd(), 'output')
    
    # --- Real progress from saved on disk ---
    if "batch_completed_files" not in st.session_state:
        st.session_state["batch_completed_files"] = {"draw": [], "threshold": [], "process": []}

    actual_progress = {"draw": [], "threshold": [], "process": []}
    if os.path.exists(output_folder):
        for folder_name in os.listdir(output_folder):
            folder_path = os.path.join(output_folder, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # draw concluded if there is dense.nii
            dense_path = os.path.join(folder_path, "dense.nii")
            if os.path.exists(dense_path):
                actual_progress["draw"].append(folder_name)
            
            # threshold concluded if there is threshold.json
            threshold_json = os.path.join(folder_path, "threshold.json")
            if os.path.exists(threshold_json):
                try:
                    with open(threshold_json, "r") as f:
                        data = json.load(f)
                        if "threshold" in data:
                            actual_progress["threshold"].append(folder_name)
                except Exception as e:
                    st.warning(f"Could not read threshold for {folder_name}: {e}")            

            # process concluded if there is tem mask.nii
            mask_path = os.path.join(folder_path, "dense_mask", "mask.nii")
            if os.path.exists(mask_path):
                actual_progress["process"].append(folder_name)         

    st.session_state["batch_completed_files"] = actual_progress

    # --- Reload thresholds disk saved thresholds ---
    st.session_state["batch_final_thresholds"] = {}
    for file_name in actual_progress["threshold"]:
        threshold_json = os.path.join(output_folder, file_name, "threshold.json")
        if os.path.exists(threshold_json):
            try:
                with open(threshold_json, "r") as f:
                    data = json.load(f)
                    if "threshold" in data:
                        st.session_state["batch_final_thresholds"][file_name] = float(data["threshold"])
            except Exception as e:
                st.warning(f"Could not read threshold for {file_name}: {e}")
    # ---------------------------------------------------

    # Get all .nii files
    all_files_inside_input = os.listdir(input_folder)
    all_nii_files = [f for f in all_files_inside_input if (f.endswith('.nii') or f.endswith('.nii.gz'))]
    all_dicom_folders = [f for f in all_files_inside_input if os.path.isdir(os.path.join(input_folder, f))]
    # check if there are .dicom files inside the folder
    all_dicom_folders = [f for f in all_dicom_folders if any(
        file.endswith('.dicom') or file.endswith('.dcm') 
        for file in os.listdir(os.path.join(input_folder, f))
    )]

    available_files = all_nii_files + all_dicom_folders
    
    if len(available_files) == 0 and len(all_dicom_folders) == 0:
        st.warning(f"No files found in {input_folder}")
        st.info("Please upload NIfTI (.nii) files or folders containing DICOM (.dicom/.dcm) files")
        return
    
    # Get already processed files
    already_done_files = actual_progress["process"]
    
    # Show file selection
    st.write("### Available Files")
    
    # Option to show only unprocessed files
    show_only_unprocessed = st.checkbox("Show only unprocessed files", value=True)
    
    if show_only_unprocessed:
        available_files = [f for f in available_files if f.split('.')[0] not in already_done_files]
    
    if len(available_files) == 0:
        st.info("No unprocessed files available. Uncheck the option above to see all files.")
        return
    
    # Multi-select for files
    selected_files = st.multiselect(
        "Select files to process:",
        available_files,
        default=available_files if len(available_files) <= 5 else available_files[:5]
    )
    
    if len(selected_files) == 0:
        st.warning("Please select at least one file to process.")
        return
    
    # Show selected files info
    #st.markdown('<div class="progress-section">', unsafe_allow_html=True)
    st.write(f"### Selected Files ({len(selected_files)})")
    for i, file in enumerate(selected_files, 1):
        file_name = file.split('.')[0]

        if file_name in actual_progress["process"]:
            status = "✅ Completed"
            color = "#28a745"
        elif file_name in actual_progress["threshold"]:
            status = "⏳ Ready for Processing"
            color = "#ffc107"
        elif file_name in actual_progress["draw"]:
            status = "⏳ Ready for Threshold"
            color = "#17a2b8"
        else:
            status = "⏳ Not Started"
            color = "#6c757d"

        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 5px; margin: 0.25rem 0; border-left: 3px solid {color};">
            <strong>{i}. {file}</strong><br>
            <span style="color: {color};">{status}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    
    st.session_state["batch_files"] = selected_files # update session while selecting files
    # Start batch processing button
    if st.button("Start Batch Processing", type="primary"):
        # Initialize batch processing session state
        #st.session_state["batch_files"] = selected_files
        st.session_state["batch_current_index"] = 0
        st.session_state["batch_step"] = "draw"  # draw, threshold, process
        st.session_state["current_step"] = "batch_draw"
        st.success(f"Batch processing started with {len(selected_files)} files!")
        st.rerun()
    
    # Show current batch info if in progress
    if "batch_files" in st.session_state:
        st.divider()
        #st.markdown('<div class="progress-section">', unsafe_allow_html=True)
        st.write("### Batch Progress")
        batch_files_without_extension = [f.split('.')[0] for f in st.session_state["batch_files"]]
        st.session_state["batch_files_without_extension"] = batch_files_without_extension
        total_files = len(batch_files_without_extension)
        
        # Draw step progress
        all_draw_completed = st.session_state["batch_completed_files"]["draw"]
        draw_completed = len([f for f in all_draw_completed if f in batch_files_without_extension])
        st.progress(draw_completed / total_files, text=f"Step 1 - Draw Masks: {draw_completed}/{total_files} completed")
        
        # Threshold step progress
        all_threshold_completed = st.session_state["batch_completed_files"]["threshold"]
        threshold_completed = len([f for f in all_threshold_completed if f in batch_files_without_extension])
        st.progress(threshold_completed / total_files, text=f"Step 2 - Set Thresholds: {threshold_completed}/{total_files} completed")
        
        # Process step progress
        all_process_completed = st.session_state["batch_completed_files"]["process"]
        process_completed = len([f for f in all_process_completed if f in batch_files_without_extension])
        st.progress(process_completed / total_files, text=f"Step 3 - Process Files: {process_completed}/{total_files} completed")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        #if st.button("🔄 Reset Batch"):
            # Clear batch-related session state
            #keys_to_remove = [key for key in st.session_state.keys() if key.startswith("batch_")]
            #for key in keys_to_remove:
                #del st.session_state[key]
            #st.session_state["current_step"] = "file_selection"
            #st.rerun()
