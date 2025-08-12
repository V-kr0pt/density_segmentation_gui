"""
SAM2 Main Processing - Entry point for SAM2 workflow
Pipeline: draw_step -> automatic SAM2 processing
"""
import streamlit as st
from sam_threshold_auto import sam_threshold_auto_step
from sam_inference import sam_inference_step
from sam_propagation import sam_propagation_step

def sam_step():
    """SAM2 Step - initializes SAM2 workflow with file selection"""
    st.header("SAM2 Powered Processing")
    st.write("Upload a NIfTI file (.nii or .nii.gz) or Dicom to start AI-powered segmentation with SAM2")
    
    # File selection for SAM2 processing
    uploaded_file = st.file_uploader(
        "Choose a .nii file", 
        type=["nii", "nii.gz"],
        help="Select a NIfTI medical image file for automatic segmentation"
    )
    
    if uploaded_file is not None:
        # Validate file size (optional - prevent very large files)
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        if file_size_mb > 500:  # 500MB limit
            st.warning(f"⚠️ Large file detected ({file_size_mb:.1f} MB). Processing may take longer.")
        
        # Save the uploaded file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        # Store in session state
        st.session_state["uploaded_file_path"] = temp_file_path
        st.session_state["uploaded_file_name"] = uploaded_file.name
        
        # Show file info
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ **File:** {uploaded_file.name}")
        with col2:
            st.info(f"📏 **Size:** {file_size_mb:.1f} MB")
        
        # Start processing button
        st.write("---")
        if st.button("🚀 Start SAM2 Processing", type="primary", use_container_width=True):
            st.session_state["current_step"] = "sam_threshold_auto"
            st.rerun()
    
    else:
        st.info("📁 Please upload a .nii file to start SAM2 processing.")
        
        # Show some helpful information
        with st.expander("ℹ️ About SAM2 Processing"):
            st.write("""
            **SAM2 (Segment Anything Model 2)** is an advanced AI model for image segmentation:
            
            🎯 **Automatic Processing Pipeline:**
            1. **Upload** your NIfTI medical image file
            2. **Auto-threshold** analysis detects high-intensity regions
            3. **SAM2 inference** generates precise segmentation masks  
            4. **Propagation** extends the mask through all slices
            
            ⚡ **Key Features:**
            - No manual drawing required
            - Intelligent bounding box detection
            - Video-like propagation through 3D volumes
            - High-quality segmentation results
            """)
        

def sam_threshold_step():
    """Redirect to the automatic threshold step"""
    return sam_threshold_auto_step()

def sam_process_step():
    """Redirect to the inference step"""
    return sam_inference_step()
