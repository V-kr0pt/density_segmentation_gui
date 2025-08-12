from file_selection_step import file_selection_step
from batch_draw_step import batch_draw_step
from batch_threshold_step import batch_threshold_step
from batch_process_step import batch_process_step
from sam_main import sam_step, sam_threshold_step, sam_process_step
from sam_draw_step import sam_draw_step
from sam_threshold_auto import sam_threshold_auto_step
from sam_inference import sam_inference_step
from sam_propagation import sam_propagation_step
import streamlit as st

# ================== Main App ==================
def main():
    st.set_page_config(layout="wide")
    st.title("Density Segmentation GUI")
    
    if "current_step" not in st.session_state:
        st.session_state["current_step"] = "mode_selection"
    
    # Mode selection
    if st.session_state["current_step"] == "mode_selection":
        st.header("Choose Processing Mode")
        
        _, col1, col2, _ = st.columns([1, 2, 2, 1])
        
        # Traditional Processing Mode
        with col1:
            st.subheader("Original Processing")
            st.write("Process multiple .nii or .dicom files:")
            st.write("• Select files you want to process")
            st.write("• Draw masks for each")
            st.write("• Set thresholds for each mask")
            st.write("• Process all files at once or one by one")

            if st.button("Start Image Selection", type="primary"):
                st.session_state["current_step"] = "file_selection"
                st.rerun()
        
        # SAM2 Processing Mode  
        with col2:
            st.subheader("SAM2 Processing")
            st.write("AI-powered segmentation with SAM2:")
            st.write("• Select .nii file")
            st.write("• Draw mask like traditional mode")
            st.write("• Set threshold for the drawn region")
            st.write("• Auto-generate bounding box")
            st.write("• SAM2 AI inference + propagation")
            
            if st.button("Start SAM2 Processing", type="primary"):
                st.session_state["current_step"] = "sam"
                st.rerun()
    
    # Traditional processing workflow (unified batch/single)
    elif st.session_state["current_step"] == "file_selection":
        file_selection_step()
    elif st.session_state["current_step"] == "batch_draw":
        batch_draw_step()
    elif st.session_state["current_step"] == "batch_threshold":
        batch_threshold_step()
    elif st.session_state["current_step"] == "batch_process":
        batch_process_step()
    
    # SAM2 processing workflow
    elif st.session_state["current_step"] == "sam":
        sam_step()
    elif st.session_state["current_step"] == "sam_draw":
        sam_draw_step()
    elif st.session_state["current_step"] == "sam_threshold":
        sam_threshold_step()
    elif st.session_state["current_step"] == "sam_threshold_auto":
        sam_threshold_auto_step()
    elif st.session_state["current_step"] == "sam_inference":
        sam_inference_step()
    elif st.session_state["current_step"] == "sam_process":
        sam_process_step()
    elif st.session_state["current_step"] == "sam_propagation":
        sam_propagation_step()
    
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
        if current_step.startswith("batch_") or current_step == "file_selection":
            st.write("**Mode:** Traditional Processing")
            if current_step == "file_selection":
                st.write("**Step:** File Selection")
            elif current_step == "batch_draw":
                st.write("**Step:** Drawing Masks")
            elif current_step == "batch_threshold":
                st.write("**Step:** Setting Thresholds")
            elif current_step == "batch_process":
                st.write("**Step:** Processing Files")
        elif current_step in ["sam", "sam_draw", "sam_threshold", "sam_threshold_auto", "sam_inference", "sam_process", "sam_propagation"]:
            st.write("**Mode:** SAM2 Processing")
            if current_step == "sam":
                st.write("**Step:** File Selection")
            elif current_step == "sam_draw":
                st.write("**Step:** Drawing Mask")
            elif current_step == "sam_threshold":
                st.write("**Step:** Setting Threshold")
            elif current_step == "sam_threshold_auto":
                st.write("**Step:** Auto Threshold Analysis")
            elif current_step == "sam_inference":
                st.write("**Step:** SAM2 Inference")
            elif current_step == "sam_process":
                st.write("**Step:** Processing")
            elif current_step == "sam_propagation":
                st.write("**Step:** Video-like Propagation")
        else:
            st.write("**Mode:** Main Menu")

if __name__ == "__main__":
    main()