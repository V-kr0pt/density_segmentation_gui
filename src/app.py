from draw_step import draw_step
from threshold_step import threshold_step
from process_step import process_step
from file_selection_step import file_selection_step
from batch_draw_step import batch_draw_step
from batch_threshold_step import batch_threshold_step
from batch_process_step import batch_process_step
import streamlit as st

# ================== Main App ==================
def main():
    st.set_page_config(layout="wide")
    st.title("Density Segmentation GUI")
    
    if "current_step" not in st.session_state:
        st.session_state["current_step"] = "mode_selection"
    
    # Mode selection
    if st.session_state["current_step"] == "mode_selection":
        #st.header("Choose Processing Mode")
        
        _, col,_ = st.columns(3)
        
        # centralized write
        with col:
            st.subheader("Process multiple .nii or .dicom files:")
            st.write("• Select files you want to process")
            st.write("• Draw masks for each")
            st.write("• Set thresholds for each mask")
            st.write("• Process all files at once or one by one")

            if st.button("Start Image Selection", type="primary"):
                st.session_state["current_step"] = "file_selection"
                st.rerun()
    
    # Batch processing workflow
    elif st.session_state["current_step"] == "file_selection":
        file_selection_step()
    elif st.session_state["current_step"] == "batch_draw":
        batch_draw_step()
    elif st.session_state["current_step"] == "batch_threshold":
        batch_threshold_step()
    elif st.session_state["current_step"] == "batch_process":
        batch_process_step()
    
    # Add a back to main menu button in the sidebar for easy navigation
    with st.sidebar:
        st.write("### Navigation")
        if st.button("🏠 Back to Main Menu"):
            # Clear all session state except for completed work
            keys_to_keep = []
            if "batch_completed_files" in st.session_state:
                # Keep batch progress if user wants to continue
                if st.checkbox("Keep batch progress"):
                    keys_to_keep = ["batch_files", "batch_completed_files", "batch_final_thresholds", "batch_thresholds"]
            
            keys_to_remove = [key for key in st.session_state.keys() if key not in keys_to_keep]
            for key in keys_to_remove:
                del st.session_state[key]
            
            st.session_state["current_step"] = "mode_selection"
            st.rerun()
        
        # Show current mode and step
        current_step = st.session_state.get("current_step", "mode_selection")
        if current_step.startswith("batch_"):
            st.write("**Mode:** Batch Processing")
            if current_step == "batch_draw":
                st.write("**Step:** Drawing Masks")
            elif current_step == "batch_threshold":
                st.write("**Step:** Setting Thresholds")
            elif current_step == "batch_process":
                st.write("**Step:** Processing Files")
        elif current_step == "file_selection":
            st.write("**Mode:** Batch Processing")
            st.write("**Step:** File Selection")
        else:
            st.write("**Mode:** Main Menu")

if __name__ == "__main__":
    main()