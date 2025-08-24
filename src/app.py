from file_selection_step import file_selection_step
from batch_draw_step import batch_draw_step
from batch_threshold_step import batch_threshold_step
from batch_process_step import batch_process_step
from batch_sam2_process_step import process_sam2_video_segmentation
import streamlit as st

# ================== Main App ==================
def main():
    st.set_page_config(
        page_title="Density Segmentation GUI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for clean styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        color: #2c3e50;
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #00b894;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(168, 237, 234, 0.3);
    }
    
    .feature-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
        color: #2d3436;
    }
    
    .feature-card p {
        margin: 0;
        font-size: 0.9rem;
        color: #2d3436;
    }
    
    .description {
        text-align: center;
        color: #6c757d;
        font-size: 1rem;
        margin: 0.5rem 0 1rem 0;
    }
    
    .compact-section {
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main title with custom styling
    st.markdown('<h1 class="main-header">Density Segmentation GUI</h1>', unsafe_allow_html=True)
    
    if "current_step" not in st.session_state:
        st.session_state["current_step"] = "mode_selection"
    
    if "processing_mode" not in st.session_state:
        st.session_state["processing_mode"] = "traditional"
    
    # Mode selection
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
            
            # Mode selection
            st.markdown("### 🎯 Select Processing Mode")
            
            mode = st.radio(
                "Choose your segmentation approach:",
                options=["traditional", "sam2"],
                format_func=lambda x: "🔧 Traditional Mode (Dynamic Thresholding)" if x == "traditional" 
                                    else "🤖 SAM2 Mode (AI Video Propagation)",
                key="mode_selection_radio"
            )
            
            if mode == "traditional":
                st.info("""
                **Traditional Mode:** Uses dynamic thresholding for slice-by-slice segmentation.
                - Fast processing
                - Parameter-based approach
                - Good for consistent density regions
                """)
            else:
                st.info("""
                **SAM2 Mode:** Uses AI-powered video propagation for advanced segmentation.
                - First slice: Threshold + SAM2 inference
                - Remaining slices: SAM2 video propagation
                - Better temporal consistency
                - Requires SAM2 installation
                """)
            
            st.session_state["processing_mode"] = mode
            
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
        # Route to appropriate processing mode
        processing_mode = st.session_state.get("processing_mode", "traditional")
        if processing_mode == "sam2":
            process_sam2_video_segmentation()
        else:
            batch_process_step()
    
    # Prettier navigation sidebar
    with st.sidebar:
        st.markdown("""
        <style>
        .nav-header {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            color: #2d3436;
            padding: 0.5rem;
            border-radius: 6px;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        .current-step {
            background: #f8f9fa;
            color: #495057;
            padding: 0.5rem;
            border-radius: 6px;
            border-left: 3px solid #00b894;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="nav-header">Navigation</div>', unsafe_allow_html=True)
        
        if st.button("🏠 Back to Main Menu", use_container_width=True):
            # Clear all session state except for completed work
            keys_to_keep = []
            if "batch_completed_files" in st.session_state:
                # Keep batch progress if user wants to continue
                if st.checkbox("📂 Keep current progress"):
                    keys_to_keep = ["batch_files", "batch_completed_files", "batch_final_thresholds", "batch_thresholds", "processing_mode"]
            
            keys_to_remove = [key for key in st.session_state.keys() if key not in keys_to_keep]
            for key in keys_to_remove:
                del st.session_state[key]
            
            st.session_state["current_step"] = "mode_selection"
            st.rerun()
        
        # Show current step with prettier styling
        current_step = st.session_state.get("current_step", "mode_selection")
        processing_mode = st.session_state.get("processing_mode", "traditional")
        
        if current_step == "file_selection":
            step_text = "📂 File Selection"
        elif current_step == "batch_draw":
            step_text = "🎨 Drawing Masks"
        elif current_step == "batch_threshold":
            step_text = "🎯 Setting Thresholds"
        elif current_step == "batch_process":
            if processing_mode == "sam2":
                step_text = "🤖 SAM2 Processing"
            else:
                step_text = "⚙️ Processing Files"
        else:
            step_text = "🏠 Main Menu"
        
        st.markdown(f'<div class="current-step"><strong>Current Step:</strong><br>{step_text}</div>', unsafe_allow_html=True)
        
        # Show processing mode
        if current_step != "mode_selection":
            mode_text = "🤖 SAM2 Mode" if processing_mode == "sam2" else "🔧 Traditional Mode"
            st.markdown(f'<div class="current-step"><strong>Mode:</strong><br>{mode_text}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()