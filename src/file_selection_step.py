import os
import streamlit as st
import json
from new_utils import resolve_dense_mask_path, resolve_final_mask_path


# =============================
# File Selection Step Function
# =============================
def file_selection_step():
    # --- UI Header and Styling ---
    st.header("📁 Step 1: Select Files")

    # Load external CSS for styling
    with open("static/file_selection_step.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Step description
    st.markdown("""
    <div class="step-container">
        <p>Choose NIfTI (.nii or .nii.gz) files or DICOM folders for batch processing.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Main input folder selection ---
    default_input = os.path.join(os.getcwd(), "media")
    selected_input = st.text_input(
        "Main Input Folder",
        value=st.session_state.get("main_input_folder", default_input),
        help="Root folder containing your files or patient subfolders (e.g., media/patient_1)"
    )

    if selected_input.strip() == "":
        st.warning("Please provide a valid input folder path.")
        return
    if not os.path.exists(selected_input):
        st.error(f"The selected folder does not exist: {selected_input}")
        return

    st.session_state["main_input_folder"] = selected_input

    input_folder = st.session_state["main_input_folder"]
    output_folder = os.path.join(os.getcwd(), 'output')

    # =============================
    # Load Progress from Disk
    # =============================
    # Initialize session state for completed files if not present
    if "batch_completed_files" not in st.session_state:
        st.session_state["batch_completed_files"] = {"draw": [], "threshold": [], "process": []}

    # Track actual progress by checking output folder contents
    actual_progress = {"draw": [], "threshold": [], "process": []}
    if os.path.exists(output_folder):
        for folder_name in os.listdir(output_folder):
            folder_path = os.path.join(output_folder, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # Draw step: completed if dense mask exists (NIfTI or DICOM)
            if resolve_dense_mask_path(folder_path) is not None:
                actual_progress["draw"].append(folder_name)

            # Threshold step: completed if threshold.json exists and contains 'threshold'
            threshold_json = os.path.join(folder_path, "threshold.json")
            if os.path.exists(threshold_json):
                try:
                    with open(threshold_json, "r") as f:
                        data = json.load(f)
                        if "threshold" in data:
                            actual_progress["threshold"].append(folder_name)
                except Exception as e:
                    st.warning(f"Could not read threshold for {folder_name}: {e}")

            # Process step: completed if final mask exists (NIfTI or DICOM)
            if resolve_final_mask_path(folder_path) is not None:
                actual_progress["process"].append(folder_name)

    st.session_state["batch_completed_files"] = actual_progress

    # =============================
    # Reload Thresholds from Disk
    # =============================
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

    # =============================
    # List Available Files
    # =============================
    # Get all files in input folder
    all_files_inside_input = os.listdir(input_folder)
    all_nii_files = [f for f in all_files_inside_input if (f.endswith('.nii') or f.endswith('.nii.gz'))]
    all_dicom_files = [f for f in all_files_inside_input if (f.endswith('.dcm') or f.endswith('.dicom'))]

    # Check for folder inside the main media folder
    all_dicom_folders = [f for f in all_files_inside_input if os.path.isdir(os.path.join(input_folder, f))]
    # Only keep folders that contain .dicom or .dcm files
    all_dicom_folders = [f for f in all_dicom_folders if any(
        file.endswith('.dicom') or file.endswith('.dcm')
        for file in os.listdir(os.path.join(input_folder, f))
    )]

    available_files = all_nii_files + all_dicom_files + all_dicom_folders

    # If no files found, show info and return
    if len(available_files) == 0 and len(all_dicom_folders) == 0:
        st.warning(f"No files found in {input_folder}")
        st.info("Please upload NIfTI (.nii) files or folders containing DICOM (.dicom/.dcm) files")
        return

    # Get already processed files (for filtering)
    already_done_files = actual_progress["process"]

    # =============================
    # File Selection UI
    # =============================
    st.write("### Available Files")

    # Checkbox: show only unprocessed files
    show_only_unprocessed = st.checkbox("Show only unprocessed files", value=True)

    if show_only_unprocessed:
        available_files = [f for f in available_files if f.split('.')[0] not in already_done_files]

    if len(available_files) == 0:
        st.info("No unprocessed files available. Uncheck the option above to see all files.")
        return

    # Multi-select widget for file selection
    selected_files = st.multiselect(
        "Select files to process:",
        available_files,
        default=available_files if len(available_files) <= 20 else available_files[:20]
    )

    if len(selected_files) == 0:
        st.warning("Please select at least one file to process.")
        return

    # =============================
    # Show Selected Files and Status
    # =============================
    st.write(f"### Selected Files ({len(selected_files)})")
    for i, file in enumerate(selected_files, 1):
        file_name = file.split('.')[0]

        # Determine status and color for each file
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

    # =============================
    # Batch Processing Controls
    # =============================
    # Update session state with selected files
    st.session_state["batch_files"] = selected_files

    # Button to start batch processing
    if st.button("Start Batch Processing", type="primary"):
        st.session_state["batch_current_index"] = 0
        st.session_state["batch_step"] = "draw"  # draw, threshold, process
        st.session_state["current_step"] = "batch_draw"
        st.success(f"Batch processing started with {len(selected_files)} files!")
        st.rerun()

    # =============================
    # Show Batch Progress (if in progress)
    # =============================
    if "batch_files" in st.session_state:
        st.divider()
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
