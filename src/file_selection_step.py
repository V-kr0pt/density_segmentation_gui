import os
import streamlit as st
import json
from new_utils import resolve_final_mask_path, has_polygon_mask


# =============================
# Helpers
# =============================
def find_dicom_series_folders(input_folder):
    """
    Recursively find every folder that directly contains DICOM (.dcm / .dicom) files.
    Returns paths relative to input_folder, using forward slashes on all platforms.
    This handles arbitrarily nested structures, e.g.:
        media/Patient_001/Series_CC/*.dcm
        media/Series_001/*.dcm
    """
    result = []
    for root, dirs, files in os.walk(input_folder):
        if any(f.lower().endswith(('.dcm', '.dicom')) for f in files):
            rel = os.path.relpath(root, input_folder).replace(os.sep, '/')
            if rel != '.':          # skip input_folder itself
                result.append(rel)
    return result


def scan_output_progress(output_folder):
    """
    Recursively walk the output folder and return three lists of relative paths
    (using forward slashes) for cases that have completed each pipeline step.
    Works for both flat layouts (output/scan1/) and nested layouts
    (output/Patient_001/Series_CC/).
    """
    progress = {"draw": [], "threshold": [], "process": []}
    if not os.path.exists(output_folder):
        return progress

    for root, dirs, files in os.walk(output_folder):
        rel = os.path.relpath(root, output_folder).replace(os.sep, '/')
        if rel == '.':
            continue

        if has_polygon_mask(root):
            progress["draw"].append(rel)

        threshold_json = os.path.join(root, "threshold.json")
        if os.path.exists(threshold_json):
            try:
                with open(threshold_json, "r") as f:
                    data = json.load(f)
                    if "threshold" in data:
                        progress["threshold"].append(rel)
            except Exception:
                pass   # handled with a warning below if needed

        if resolve_final_mask_path(root) is not None:
            progress["process"].append(rel)

    return progress


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

    # Track actual progress by checking output folder contents (recursive)
    actual_progress = scan_output_progress(output_folder)

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

    # Check for DICOM series folders at any depth inside the input folder
    all_dicom_folders = find_dicom_series_folders(input_folder)

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
        default=available_files if len(available_files) <= 300 else available_files[:300]
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