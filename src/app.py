from file_selection_step import file_selection_step
from batch_draw_step import batch_draw_step
from batch_threshold_step import batch_threshold_step
from batch_process_step import batch_process_step
from advanced_pipeline_step import advanced_pipeline_step
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
                <h4>Processing Modes</h4>
                <p><strong>🎯 Standard Pipeline:</strong> Traditional threshold-based segmentation with manual drawing</p>
                <p><strong>🚀 Advanced SAM2 Pipeline:</strong> AI-powered segmentation with center-outward propagation and adaptive refinement</p>
                <hr>
                <h4>How it works</h4>
                <p><strong>1. Select files:</strong> Choose your NIfTI (.nii or .nii.gz) or DICOM files from the media directory</p>
                <p><strong>2. Draw masks:</strong> Create interactive masks for each image</p>
                <p><strong>3. Set thresholds:</strong> Adjust parameters for optimal segmentation</p>
                <p><strong>4. Process:</strong> Generate results with batch processing</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add spacing
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Mode selection buttons
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("🎯 Standard Pipeline", type="secondary", use_container_width=True):
                    st.session_state["processing_mode"] = "standard"
                    st.session_state["current_step"] = "file_selection"
                    st.rerun()
            
            with col_b:
                if st.button("🚀 Advanced SAM2 Pipeline", type="primary", use_container_width=True):
                    st.session_state["processing_mode"] = "advanced"
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
    elif st.session_state["current_step"] == "advanced_pipeline":
        advanced_pipeline_step()
    
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
        processing_mode = st.session_state.get("processing_mode", "standard")
        
        if current_step == "file_selection":
            step_text = "📂 File Selection"
        elif current_step == "batch_draw":
            step_text = "🎨 Drawing Masks"
        elif current_step == "batch_threshold":
            step_text = "🎯 Setting Thresholds"
        elif current_step == "batch_process":
            step_text = "⚙️ Processing Files"
        elif current_step == "advanced_pipeline":
            step_text = "🚀 Advanced SAM2 Pipeline"
        else:
            step_text = "🏠 Main Menu"
        
        # Show processing mode
        mode_text = "🚀 Advanced SAM2" if processing_mode == "advanced" else "🎯 Standard"
        
        st.markdown(f'<div class="current-step"><strong>Mode:</strong> {mode_text}<br><strong>Current Step:</strong><br>{step_text}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()