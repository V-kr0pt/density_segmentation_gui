"""
SAM2 Automatic Threshold Detection and Bounding Box Generation
"""
import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from utils import ImageOperations, ThresholdOperations
from sam_utils import SAM2Manager, convert_nii_slice_for_sam2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import io

def find_bounding_box(mask, padding=10):
    """
    Find tight bounding box around mask region
    
    Args:
        mask: Binary mask array
        padding: Extra padding around bounding box
    
    Returns:
        [x_min, y_min, x_max, y_max] in format expected by SAM2
    """
    # Find all non-zero points
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Add padding
    height, width = mask.shape
    x_min = max(0, cmin - padding)
    y_min = max(0, rmin - padding)
    x_max = min(width, cmax + padding)
    y_max = min(height, rmax + padding)
    
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

def visualize_threshold_and_bbox(image, mask, threshold, bbox):
    """
    Create visualization showing original image, thresholded mask, and bounding box
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Thresholded mask
    thresholded_mask = (mask > threshold).astype(np.uint8)
    axes[1].imshow(image, cmap='gray', alpha=0.7)
    axes[1].imshow(thresholded_mask, cmap='Reds', alpha=0.5)
    axes[1].set_title(f'Threshold: {threshold:.3f}')
    axes[1].axis('off')
    
    # Image with bounding box
    axes[2].imshow(image, cmap='gray')
    if bbox is not None:
        rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                        linewidth=2, edgecolor='red', facecolor='none')
        axes[2].add_patch(rect)
        axes[2].set_title('Auto-Generated Bounding Box')
    else:
        axes[2].set_title('No Region Detected')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def sam_threshold_auto_step():
    """
    Automatic threshold detection and bounding box generation step
    """
    st.header("SAM2 Processing - Automatic Threshold Detection")
    
    if "sam_selected_file" not in st.session_state:
        st.error("No file selected. Please go back to file selection.")
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "sam"
            st.rerun()
        return
    
    file_path = st.session_state["sam_file_path"]
    file_name = st.session_state["sam_file_name"]
    
    st.write(f"### Processing: `{st.session_state['sam_selected_file']}`")
    
    try:
        # Load the image and use the saved mask from draw step
        image, affine, nb_of_slices = ImageOperations.load_image(file_path)
        
        # Load the mask created in the draw step
        output_path = os.path.join(os.getcwd(), 'output', file_name)
        mask_path = os.path.join(output_path, 'dense.nii')
        
        if not os.path.exists(mask_path):
            st.error("No mask found. Please complete the draw step first.")
            if st.button("← Back to Draw Step"):
                st.session_state["current_step"] = "sam_draw"
                st.rerun()
            return
        
        # Load the saved threshold
        threshold_json = os.path.join(output_path, "threshold.json")
        auto_threshold = 0.45  # Default fallback
        
        if os.path.exists(threshold_json):
            try:
                import json
                with open(threshold_json, "r") as f:
                    data = json.load(f)
                    auto_threshold = data.get("threshold", 0.45)
                st.success(f"Using saved threshold: {auto_threshold:.3f}")
            except:
                st.warning("Could not load saved threshold, using default: 0.45")
        else:
            st.warning("No saved threshold found, using default: 0.45")
        
        st.info(f"🔄 Using threshold {auto_threshold:.3f} for bounding box detection...")
        
        # Load central slice and mask
        central_slice = ImageOperations.load_nii_central_slice(file_path)
        mask_slice = ImageOperations.load_nii_central_slice(mask_path, flip=True)
        
        # Apply threshold to the mask (not to the original image)
        binary_mask = (mask_slice > auto_threshold).astype(np.uint8)
        
        # Find bounding box
        bbox = find_bounding_box(binary_mask, padding=20)
        
        if bbox is None:
            st.error("No region detected with the current threshold. The drawn mask may be too small or the threshold too high.")
            if st.button("← Back to Threshold Step"):
                st.session_state["current_step"] = "sam_threshold"
                st.rerun()
            return
        
        # Store results
        st.session_state["sam_threshold"] = auto_threshold
        st.session_state["sam_central_slice"] = central_slice
        st.session_state["sam_binary_mask"] = binary_mask
        st.session_state["sam_bounding_box"] = bbox
        st.session_state["sam_image_data"] = image
        st.session_state["sam_affine"] = affine
        st.session_state["sam_nb_slices"] = nb_of_slices
        
        # Display results
        fig = visualize_threshold_and_bbox(central_slice, mask_slice, auto_threshold, bbox)
        
        st.subheader("Automatic Analysis Results")
        st.pyplot(fig)
        
        # Show bounding box coordinates
        st.write("**Detected Bounding Box:**")
        st.write(f"- Top-left: ({bbox[0]:.0f}, {bbox[1]:.0f})")
        st.write(f"- Bottom-right: ({bbox[2]:.0f}, {bbox[3]:.0f})")
        st.write(f"- Width: {bbox[2] - bbox[0]:.0f} pixels")
        st.write(f"- Height: {bbox[3] - bbox[1]:.0f} pixels")
        
        # Continue button
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("← Back to Threshold Step"):
                st.session_state["current_step"] = "sam_threshold"
                st.rerun()
        
        with col2:
            if st.button("✅ Continue to SAM2 Inference", type="primary"):
                st.session_state["current_step"] = "sam_inference"
                st.rerun()
        
        with col3:
            if st.button("🔄 Adjust Threshold"):
                st.session_state["current_step"] = "sam_threshold"
                st.rerun()
        
    except Exception as e:
        st.error(f"Error in automatic threshold detection: {str(e)}")
        if st.button("← Back to Threshold Step"):
            st.session_state["current_step"] = "sam_threshold"
            st.rerun()
