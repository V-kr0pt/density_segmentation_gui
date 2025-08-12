"""
SAM2 Video-like Propagation Step - Propagate segmentation across all slices
"""
import os
import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import shutil
from sam_utils import SAM2Manager, convert_nii_slice_for_sam2
from utils import ImageOperations
import matplotlib.pyplot as plt

try:
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    SAM2_VIDEO_AVAILABLE = True
except ImportError:
    SAM2_VIDEO_AVAILABLE = False

def prepare_video_frames(image_data, temp_dir):
    """
    Convert 3D medical image to individual frame files for SAM2 video processing
    
    Args:
        image_data: 3D numpy array (slices, height, width)
        temp_dir: Temporary directory to save frames
    
    Returns:
        frame_names: List of frame filenames
    """
    frame_names = []
    
    for slice_idx in range(image_data.shape[2]):  # Assuming slices are the 3rd dimension
        # Extract slice
        slice_2d = image_data[:, :, slice_idx]
        
        # Convert to SAM2 format
        sam2_frame = convert_nii_slice_for_sam2(slice_2d)
        
        # Save as image file
        frame_name = f"frame_{slice_idx:04d}.jpg"
        frame_path = os.path.join(temp_dir, frame_name)
        
        pil_image = Image.fromarray(sam2_frame)
        pil_image.save(frame_path)
        
        frame_names.append(frame_name)
    
    return frame_names

def visualize_propagation_results(video_segments, frame_names, sample_frames=5):
    """
    Visualize propagation results for a sample of frames
    """
    if not video_segments:
        return None
    
    # Select frames to visualize
    total_frames = len(frame_names)
    if total_frames <= sample_frames:
        selected_frames = list(range(total_frames))
    else:
        stride = max(1, total_frames // sample_frames)
        selected_frames = list(range(0, total_frames, stride))[:sample_frames]
    
    fig, axes = plt.subplots(1, len(selected_frames), figsize=(15, 3))
    if len(selected_frames) == 1:
        axes = [axes]
    
    for i, frame_idx in enumerate(selected_frames):
        if frame_idx in video_segments:
            # Load original frame
            frame_path = os.path.join(st.session_state["sam_temp_dir"], frame_names[frame_idx])
            frame_img = np.array(Image.open(frame_path))
            
            axes[i].imshow(frame_img)
            
            # Overlay masks
            for obj_id, mask in video_segments[frame_idx].items():
                axes[i].imshow(mask, alpha=0.5, cmap='Reds')
            
            axes[i].set_title(f'Frame {frame_idx}')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f'Frame {frame_idx}\n(no data)', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def sam_propagation_step():
    """
    SAM2 video-like propagation across all slices
    """
    st.header("SAM2 Processing - Video-like Propagation")
    
    # Check prerequisites
    if "sam_best_mask" not in st.session_state:
        st.error("No SAM2 inference results found. Please complete the inference step first.")
        if st.button("← Back to Inference"):
            st.session_state["current_step"] = "sam_inference"
            st.rerun()
        return
    
    if not SAM2_VIDEO_AVAILABLE:
        st.error("SAM2 video predictor not available. Please install the complete SAM2 package.")
        st.code("pip install git+https://github.com/facebookresearch/segment-anything-2.git")
        return
    
    file_name = st.session_state["sam_file_name"]
    image_data = st.session_state["sam_image_data"]
    best_mask = st.session_state["sam_best_mask"]
    bbox = st.session_state["sam_bounding_box"]
    inference_frame_idx = st.session_state.get("sam_inference_frame", 0)
    
    st.write(f"### Propagating segmentation across all slices: `{st.session_state['sam_selected_file']}`")
    st.write(f"**Total slices to process:** {image_data.shape[2]}")
    
    # Create temporary directory for video frames
    if "sam_temp_dir" not in st.session_state:
        st.session_state["sam_temp_dir"] = tempfile.mkdtemp()
    
    temp_dir = st.session_state["sam_temp_dir"]
    
    try:
        # Prepare video frames
        with st.spinner("Preparing video frames from medical image slices..."):
            frame_names = prepare_video_frames(image_data, temp_dir)
            st.session_state["sam_frame_names"] = frame_names
            st.success(f"Prepared {len(frame_names)} frames")
        
        # Initialize video predictor
        with st.spinner("Initializing SAM2 video predictor..."):
            # Try different config options for video predictor
            config_options = [
                "sam2.1_hiera_l.yaml",
                "configs/sam2.1_hiera_l.yaml",
                "sam2.1/sam2.1_hiera_l.yaml"
            ]
            
            checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
            predictor = None
            
            for config in config_options:
                try:
                    predictor = build_sam2_video_predictor(config, checkpoint_path)
                    break
                except Exception as e:
                    continue
            
            if predictor is None:
                st.error("Failed to initialize SAM2 video predictor")
                return
            
            st.success("Video predictor initialized")
        
        # Run propagation
        with st.spinner("Running SAM2 propagation across all slices..."):
            # Initialize inference state
            inference_state = predictor.init_state(video_path=temp_dir)
            predictor.reset_state(inference_state)
            
            # Add bounding box to the inference frame (central slice)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=inference_frame_idx,
                obj_id=0,  # First object
                box=bbox,
            )
            
            # Run propagation throughout the video
            video_segments = {}
            progress_bar = st.progress(0)
            frame_count = 0
            
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                frame_count += 1
                progress_bar.progress(frame_count / len(frame_names))
            
            st.session_state["sam_video_segments"] = video_segments
            st.session_state["sam_inference_state"] = inference_state
            st.success(f"Propagation completed! Processed {len(video_segments)} frames")
        
        # Display results
        st.subheader("Propagation Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Frames", len(frame_names))
            st.metric("Processed Frames", len(video_segments))
        with col2:
            st.metric("Objects per Frame", len(out_obj_ids))
            coverage = len(video_segments) / len(frame_names) * 100
            st.metric("Coverage", f"{coverage:.1f}%")
        
        # Visualize sample results
        st.subheader("Sample Propagation Results")
        fig = visualize_propagation_results(video_segments, frame_names)
        if fig:
            st.pyplot(fig)
        
        # Show processing summary
        with st.expander("Detailed Results"):
            st.write("**Frame-by-frame summary:**")
            for frame_idx in sorted(list(video_segments.keys())[:10]):  # Show first 10
                masks = video_segments[frame_idx]
                st.write(f"Frame {frame_idx}: {len(masks)} objects")
                for obj_id, mask in masks.items():
                    coverage = np.sum(mask) / mask.size * 100
                    st.write(f"  Object {obj_id}: {coverage:.1f}% coverage")
            
            if len(video_segments) > 10:
                st.write(f"... and {len(video_segments) - 10} more frames")
        
        # Navigation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("← Back to Inference"):
                st.session_state["current_step"] = "sam_inference"
                st.rerun()
        
        with col2:
            if st.button("🔄 Re-run Propagation"):
                # Clear results and re-run
                keys_to_clear = ["sam_video_segments", "sam_inference_state"]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        with col3:
            if st.button("✅ Complete Processing", type="primary"):
                st.session_state["current_step"] = "sam_process"
                st.rerun()
                
    except Exception as e:
        st.error(f"Error during propagation: {str(e)}")
        st.exception(e)
        
    finally:
        # Cleanup temporary directory on session end
        if st.button("🧹 Clean Temporary Files"):
            if "sam_temp_dir" in st.session_state and os.path.exists(st.session_state["sam_temp_dir"]):
                shutil.rmtree(st.session_state["sam_temp_dir"])
                del st.session_state["sam_temp_dir"]
                st.success("Temporary files cleaned up")
