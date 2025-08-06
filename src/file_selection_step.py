import os
import streamlit as st

def file_selection_step():
    st.header("Batch File Selection")
    st.write("Select the .nii files you want to process in batch.")
    
    input_folder = os.path.join(os.getcwd(), 'media')
    output_folder = os.path.join(os.getcwd(), 'output')
    
    # Get all .nii files
    available_files = [f for f in os.listdir(input_folder) if f.endswith('.nii')]
    
    if len(available_files) == 0:
        st.warning(f"No .nii files found in {input_folder}")
        return
    
    # Get already processed files
    already_done_files = []
    if os.path.exists(output_folder):
        already_done_files = os.listdir(output_folder)
    
    # Show file selection
    st.write("### Available Files:")
    
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
    st.write(f"### Selected {len(selected_files)} files:")
    for i, file in enumerate(selected_files, 1):
        status = "✅ Processed" if file.split('.')[0] in already_done_files else "⏳ Pending"
        st.write(f"{i}. `{file}` - {status}")
    
    # Start batch processing button
    if st.button("🚀 Start Batch Processing", type="primary"):
        # Initialize batch processing session state
        st.session_state["batch_files"] = selected_files
        st.session_state["batch_current_index"] = 0
        st.session_state["batch_step"] = "draw"  # draw, threshold, process
        st.session_state["batch_completed_files"] = {
            "draw": [],
            "threshold": [],
            "process": []
        }
        st.session_state["current_step"] = "batch_draw"
        st.success(f"Batch processing started with {len(selected_files)} files!")
        st.rerun()
    
    # Show current batch info if in progress
    if "batch_files" in st.session_state:
        st.divider()
        st.write("### Current Batch Progress:")
        total_files = len(st.session_state["batch_files"])
        
        # Draw step progress
        draw_completed = len(st.session_state["batch_completed_files"]["draw"])
        st.progress(draw_completed / total_files, text=f"Draw Step: {draw_completed}/{total_files} completed")
        
        # Threshold step progress
        threshold_completed = len(st.session_state["batch_completed_files"]["threshold"])
        st.progress(threshold_completed / total_files, text=f"Threshold Step: {threshold_completed}/{total_files} completed")
        
        # Process step progress
        process_completed = len(st.session_state["batch_completed_files"]["process"])
        st.progress(process_completed / total_files, text=f"Process Step: {process_completed}/{total_files} completed")
        
        if st.button("🔄 Reset Batch"):
            # Clear batch-related session state
            keys_to_remove = [key for key in st.session_state.keys() if key.startswith("batch_")]
            for key in keys_to_remove:
                del st.session_state[key]
            st.session_state["current_step"] = "file_selection"
            st.rerun()
