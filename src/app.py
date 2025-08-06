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
    st.title("# Density Segmentation GUI")
    
    if "current_step" not in st.session_state:
        st.session_state["current_step"] = "mode_selection"
    
    # Mode selection
    if st.session_state["current_step"] == "mode_selection":
        st.header("Choose Processing Mode")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔄 Batch Processing")
            st.write("Process multiple .nii files in sequence:")
            st.write("• Select multiple files")
            st.write("• Draw masks for each file")
            st.write("• Set thresholds for each file")
            st.write("• Process all files at once")
            
            if st.button("Start Batch Processing", type="primary"):
                st.session_state["current_step"] = "file_selection"
                st.rerun()
        
        with col2:
            st.subheader("📄 Single File Processing")
            st.write("Process one file at a time (original mode):")
            st.write("• Select one file")
            st.write("• Draw mask")
            st.write("• Set threshold")
            st.write("• Process file")
            
            if st.button("Start Single File Processing"):
                st.session_state["current_step"] = "draw"
                st.rerun()
    
    # Single file processing (original workflow)
    elif st.session_state["current_step"] == "draw":
        draw_step()
    elif st.session_state["current_step"] == "threshold":
        threshold_step()
    elif st.session_state["current_step"] == "process":
        process_step()
    
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
        elif current_step in ["draw", "threshold", "process"]:
            st.write("**Mode:** Single File")
            if current_step == "draw":
                st.write("**Step:** Drawing Mask")
            elif current_step == "threshold":
                st.write("**Step:** Setting Threshold")
            elif current_step == "process":
                st.write("**Step:** Processing File")
        elif current_step == "file_selection":
            st.write("**Mode:** Batch Processing")
            st.write("**Step:** File Selection")
        else:
            st.write("**Mode:** Main Menu")

if __name__ == "__main__":
    main()