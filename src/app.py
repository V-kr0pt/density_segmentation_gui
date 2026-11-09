import os
from env_check import run_all_checks

run_all_checks()

from file_selection_step import file_selection_step
from batch_draw_step import batch_draw_step
from batch_threshold_step import batch_threshold_step
from batch_process_step import batch_process_step
from new_utils import list_output_artifacts, delete_output_artifacts
import streamlit as st

# ================== Main App ==================
def main():
    st.set_page_config(
        page_title="Density Segmentation GUI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load external CSS for styling
    with open("static/app.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Main title with custom styling
    st.markdown('<h1 class="main-header">Density Segmentation GUI</h1>', unsafe_allow_html=True)
    
    if "current_step" not in st.session_state:
        st.session_state["current_step"] = "mode_selection"
    
    def _clear_case_from_session(case_name: str, cleared: set):
        if "batch_completed_files" in st.session_state:
            if "draw" in st.session_state["batch_completed_files"] and (
                "dense.nii" in cleared or "dense.dcm" in cleared or "dense_dcm" in cleared
            ):
                st.session_state["batch_completed_files"]["draw"] = [
                    f for f in st.session_state["batch_completed_files"]["draw"] if f != case_name
                ]

            if "threshold" in st.session_state["batch_completed_files"] and "threshold.json" in cleared:
                st.session_state["batch_completed_files"]["threshold"] = [
                    f for f in st.session_state["batch_completed_files"]["threshold"] if f != case_name
                ]

            if "process" in st.session_state["batch_completed_files"] and (
                "dense_mask" in cleared or "mask.dcm" in cleared or "mask_dcm" in cleared
            ):
                st.session_state["batch_completed_files"]["process"] = [
                    f for f in st.session_state["batch_completed_files"]["process"] if f != case_name
                ]

        if "threshold.json" in cleared:
            if "batch_final_thresholds" in st.session_state:
                st.session_state["batch_final_thresholds"].pop(case_name, None)
            if "batch_thresholds" in st.session_state:
                st.session_state["batch_thresholds"].pop(case_name, None)
    
    # Sidebar navigation
    with st.sidebar:
        
        st.markdown('<div class="nav-header">Navigation</div>', unsafe_allow_html=True)
        
        if st.button("🏠 Back to Main Menu", use_container_width=True):
            # Clear all session state except for completed work
            keys_to_keep = []
            if "batch_completed_files" in st.session_state:
                # Keep batch progress if user wants to continue
                if st.checkbox("📂 Keep current progress"):
                    keys_to_keep = ["batch_files", "batch_completed_files", "batch_final_thresholds", "batch_thresholds"]
            
            keys_to_remove = [key for key in st.session_state.keys() if key not in keys_to_keep]
            for key in keys_to_remove:
                del st.session_state[key]
            
            st.session_state["current_step"] = "mode_selection"
            st.rerun()
        
        # Show current step 
        current_step = st.session_state.get("current_step", "mode_selection")
        if current_step == "file_selection":
            step_text = "📂 File Selection"
        elif current_step == "batch_draw":
            step_text = "🎨 Drawing Masks"
        elif current_step == "batch_threshold":
            step_text = "🎯 Setting Thresholds"
        elif current_step == "batch_process":
            step_text = "⚙️ Processing Files"
        else:
            step_text = "🏠 Main Menu"
        
        st.markdown(f'<div class="current-step"><strong>Current Step:</strong><br>{step_text}</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="nav-header">Manage Outputs</div>', unsafe_allow_html=True)

        output_root = os.path.join(os.getcwd(), "output")
        if not os.path.isdir(output_root):
            st.caption("No output folder found yet.")
        else:
            case_folders = sorted(
                d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))
            )
            if not case_folders:
                st.caption("No output cases found.")
            else:
                for case_name in case_folders:
                    case_path = os.path.join(output_root, case_name)
                    artifacts = list_output_artifacts(case_path)
                    if not artifacts:
                        continue
                    with st.expander(f"🧹 {case_name} ({len(artifacts)})", expanded=False):
                        for artifact in artifacts:
                            display_name = artifact["name"]
                            if artifact["kind"] == "dir":
                                display_name = f"{display_name}/"
                            col1, col2 = st.columns([8, 1], gap="small")
                            with col1:
                                st.markdown(f"`{display_name}`")
                            with col2:
                                if st.button(
                                    "🗑",
                                    key=f"delete_{case_name}_{artifact['name']}",
                                    help=f"Remove {artifact['name']} from {case_name}",
                                ):
                                    result = delete_output_artifacts(case_path, [artifact["name"]])
                                    cleared = {artifact["name"]}
                                    _clear_case_from_session(case_name, cleared)
                                    if result["errors"]:
                                        st.error("; ".join(result["errors"]))
                                    else:
                                        st.success(f"Removed {artifact['name']} from {case_name}")
                                    st.rerun()

    # Mode selection step
    if st.session_state["current_step"] == "mode_selection":
    # Welcome section
        st.markdown("""
        <div class="description">
            Interactive medical image segmentation for NIfTI and DICOM files
        </div>
        """, unsafe_allow_html=True)
        
    # Main content
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Feature overview
            st.markdown("""
            <div class="feature-card">
                <h4>How it works</h4>
                <p><strong>1. Select files:</strong> Choose your NIfTI (.nii or .nii.gz) or DICOM files from the media directory</p>
                <p><strong>2. Draw masks:</strong> Create interactive masks for each image</p>
                <p><strong>3. Set thresholds:</strong> Adjust parameters for optimal segmentation</p>
                <p><strong>4. Process:</strong> Generate results with batch processing</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add spacing
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Start button
            if st.button("Start Processing", type="primary", use_container_width=True):
                st.session_state["current_step"] = "file_selection"
                st.rerun()
            
            # Footer info
            st.markdown("""
            <div style="text-align: center; padding: 1rem 0; color: #888; font-size: 0.9rem;">
                Place your .nii or DICOM files in the media/ directory before starting
            </div>
            """, unsafe_allow_html=True)
    

    # Batch processing workflow
    elif st.session_state["current_step"] == "file_selection":
        file_selection_step()
    elif st.session_state["current_step"] == "batch_draw":
        batch_draw_step()
    elif st.session_state["current_step"] == "batch_threshold":
        batch_threshold_step()
    elif st.session_state["current_step"] == "batch_process":
        batch_process_step()

if __name__ == "__main__":
    main()
