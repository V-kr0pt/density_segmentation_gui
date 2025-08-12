"""
SAM2 Inference Step - Generate segmentation using bounding box prompt
"""
import os
import streamlit as st
import numpy as np
from PIL import Image
from sam_utils import SAM2Manager, convert_nii_slice_for_sam2, convert_nii_slice_with_threshold_overlay_for_sam2
import matplotlib.pyplot as plt
import cv2

def visualize_sam2_results(image, masks, scores, bbox, threshold_overlay_image=None):
    """
    Visualize SAM2 inference results
    """
    fig, axes = plt.subplots(1, min(4, len(masks) + 1), figsize=(15, 4))
    if len(masks) == 1:
        axes = [axes]
    
    # Show the threshold overlay image if provided (what SAM2 actually sees)
    if threshold_overlay_image is not None:
        axes[0].imshow(threshold_overlay_image)
        axes[0].set_title('SAM2 Input (with threshold overlay)')
    else:
        # Fallback to original image with bounding box
        axes[0].imshow(image, cmap='gray')
        if bbox is not None:
            from matplotlib.patches import Rectangle
            rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                            linewidth=2, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
        axes[0].set_title('Input + Bounding Box')
    
    # Add bounding box to threshold overlay if provided
    if threshold_overlay_image is not None and bbox is not None:
        from matplotlib.patches import Rectangle
        rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                        linewidth=2, edgecolor='yellow', facecolor='none')
        axes[0].add_patch(rect)
    
    axes[0].axis('off')
    
    # Show up to 3 best masks
    for i in range(min(3, len(masks))):
        if i + 1 < len(axes):
            axes[i + 1].imshow(image, cmap='gray', alpha=0.7)
            axes[i + 1].imshow(masks[i], cmap='Reds', alpha=0.5)
            axes[i + 1].set_title(f'Mask {i+1} (Score: {scores[i]:.3f})')
            axes[i + 1].axis('off')
    
    plt.tight_layout()
    return fig

def sam_inference_step():
    """
    SAM2 inference step using the detected bounding box
    """
    st.header("SAM2 Processing - AI Inference")
    
    # Check prerequisites
    if "sam_bounding_box" not in st.session_state:
        st.error("No bounding box detected. Please complete the threshold detection step first.")
        if st.button("← Back to Threshold Detection"):
            st.session_state["current_step"] = "sam_threshold_auto"
            st.rerun()
        return
    
    file_name = st.session_state["sam_file_name"]
    central_slice = st.session_state["sam_central_slice"]
    mask_slice = st.session_state.get("sam_mask_slice")
    threshold = st.session_state.get("sam_threshold", 0.45)
    bbox = st.session_state["sam_bounding_box"]
    
    if mask_slice is None:
        st.error("Mask slice not found. Please complete the threshold detection step first.")
        if st.button("← Back to Threshold Detection"):
            st.session_state["current_step"] = "sam_threshold_auto"
            st.rerun()
        return
    
    st.write(f"### Running SAM2 inference on: `{st.session_state['sam_selected_file']}`")
    
    # Initialize SAM2
    with st.spinner("Initializing SAM2 model..."):
        sam_manager = SAM2Manager()
        
        # Load model
        success, message = sam_manager.load_model()
        if not success:
            st.error(f"Failed to load SAM2 model: {message}")
            return
        
        st.success(message)
    
    # Prepare image for SAM2
    with st.spinner("Preparing image with threshold overlay for inference..."):
        # Use the new function that combines original image with threshold overlay
        sam2_image = convert_nii_slice_with_threshold_overlay_for_sam2(
            central_slice, mask_slice, threshold, overlay_alpha=0.3
        )
        
        # Set image in predictor
        success, message = sam_manager.set_image(sam2_image)
        if not success:
            st.error(f"Failed to set image: {message}")
            return
    
    # Run inference
    with st.spinner("Running SAM2 inference..."):
        masks, scores, message = sam_manager.predict(input_boxes=bbox.reshape(1, -1))
        
        if masks is None:
            st.error(f"SAM2 inference failed: {message}")
            return
        
        st.success(f"SAM2 inference completed: {message}")
    
    # Display results
    st.subheader("SAM2 Inference Results")
    
    fig = visualize_sam2_results(central_slice, masks, scores, bbox, threshold_overlay_image=sam2_image)
    st.pyplot(fig)
    
    # Show inference details
    st.write("**Inference Details:**")
    st.write(f"- Input method: Original image with threshold overlay (threshold: {threshold:.3f})")
    st.write(f"- Number of masks generated: {len(masks)}")
    st.write(f"- Best mask score: {scores[0]:.3f}")
    st.write(f"- Input bounding box: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
    
    # Select best mask
    best_mask_idx = 0  # Usually the first mask is the best
    selected_mask = masks[best_mask_idx]
    
    st.write(f"**Selected mask:** Mask {best_mask_idx + 1} (Score: {scores[best_mask_idx]:.3f})")
    
    # Store results for propagation step
    st.session_state["sam_best_mask"] = selected_mask
    st.session_state["sam_all_masks"] = masks
    st.session_state["sam_scores"] = scores
    st.session_state["sam_inference_frame"] = 0  # Central slice is frame 0
    st.session_state["sam_prepared_image"] = sam2_image
    
    # Navigation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("← Back to Threshold Detection"):
            st.session_state["current_step"] = "sam_threshold_auto"
            st.rerun()
    
    with col2:
        if st.button("🔄 Re-run Inference"):
            # Clear cached results and re-run
            keys_to_clear = ["sam_best_mask", "sam_all_masks", "sam_scores", "sam_prepared_image"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    with col3:
        if st.button("➡️ Continue to Propagation", type="primary"):
            st.session_state["current_step"] = "sam_propagation"
            st.rerun()
    
    # Advanced options
    with st.expander("Advanced Options"):
        st.write("**Mask Selection:**")
        if len(masks) > 1:
            mask_choice = st.selectbox(
                "Choose different mask:",
                options=range(len(masks)),
                format_func=lambda x: f"Mask {x+1} (Score: {scores[x]:.3f})",
                index=best_mask_idx
            )
            
            if mask_choice != best_mask_idx:
                st.session_state["sam_best_mask"] = masks[mask_choice]
                st.write(f"Selected mask {mask_choice + 1} for propagation.")
        
        st.write("**Technical Details:**")
        st.write(f"- Device: {sam_manager.device}")
        st.write(f"- Image shape: {sam2_image.shape}")
        st.write(f"- Mask shape: {selected_mask.shape}")
        st.write(f"- Mask coverage: {np.sum(selected_mask)}/{selected_mask.size} pixels ({100*np.sum(selected_mask)/selected_mask.size:.1f}%)")
