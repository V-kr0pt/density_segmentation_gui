import os
import numpy as np
import streamlit as st
import nibabel as nib
from PIL import Image
import shutil
import cv2
import torch
from utils import ImageOperations, MaskOperations, ThresholdOperations
from sam2_simple_approach import (
    create_simple_initial_mask, 
    safe_roi_extraction, 
    safe_normalize_for_sam2, 
    apply_safe_threshold
)

# SAM2 Video Predictor imports with fallback
try:
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    SAM2_VIDEO_AVAILABLE = True
except ImportError:
    SAM2_VIDEO_AVAILABLE = False

class SAM2VideoManager:
    """Manager class for SAM2 video operations with propagation"""
    
    def __init__(self):
        self.video_predictor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        self.inference_state = None
    
    def load_video_model(self):
        """Load the SAM2 video predictor model"""
        if self.model_loaded:
            return True, "Video model already loaded"
        
        if not SAM2_VIDEO_AVAILABLE:
            return False, "SAM2 video predictor not available. Install SAM2 properly."
        
        try:
            # Try different config approaches for SAM2 video
            config_options = [
                "sam2.1_hiera_l.yaml",
                "configs/sam2.1_hiera_l.yaml",
                "sam2.1/sam2.1_hiera_l.yaml",
                "sam2_hiera_l.yaml",
                "sam2.1_hiera_large.yaml",
            ]
            
            checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
            video_predictor = None
            config_used = None
            last_error = None
            
            for config_name in config_options:
                try:
                    video_predictor = build_sam2_video_predictor(config_name, checkpoint_path, device=self.device)
                    config_used = config_name
                    break
                except Exception as config_error:
                    last_error = str(config_error)
                    continue
            
            if video_predictor is None:
                return False, f"Failed to load video predictor with all configs. Last error: {last_error}"
            
            self.video_predictor = video_predictor
            self.model_loaded = True
            return True, f"SAM2 video predictor loaded successfully on device: {self.device} using config: {config_used}"
            
        except Exception as e:
            return False, f"Unexpected error loading video model: {str(e)}"
    
    def init_inference_state(self, video_frames):
        """Initialize inference state for video prediction"""
        if not self.model_loaded:
            return False, "Video model not loaded"
        
        try:
            self.inference_state = self.video_predictor.init_state(video_frames)
            return True, "Inference state initialized successfully"
        except Exception as e:
            return False, f"Error initializing inference state: {str(e)}"
    
    def add_first_frame_mask(self, frame_idx, mask, obj_id=1):
        """Add mask for the first frame to guide propagation"""
        if self.inference_state is None:
            return False, "Inference state not initialized"
        
        try:
            self.video_predictor.add_new_mask(self.inference_state, frame_idx, obj_id, mask)
            return True, "First frame mask added successfully"
        except Exception as e:
            return False, f"Error adding first frame mask: {str(e)}"
    
    def propagate_masks(self):
        """Propagate masks through video frames"""
        if self.inference_state is None:
            return None, "Inference state not initialized"
        
        try:
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(self.inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            return video_segments, "Propagation completed successfully"
        except Exception as e:
            return None, f"Error during propagation: {str(e)}"

def safe_normalize_image(img):
    """
    Safely normalize image to 0-255 range, handling edge cases
    """
    if img.size == 0:
        return np.array([], dtype=np.uint8)
    
    img_min = np.min(img)
    img_max = np.max(img)
    
    if img_max == img_min:
        # Handle case where all values are the same
        return np.zeros_like(img, dtype=np.uint8)
    
    # Normalize to 0-1 range then scale to 0-255
    img_norm = (img - img_min) / (img_max - img_min + 1e-8)
    return (img_norm * 255).astype(np.uint8)

def process_nifti_with_sam2_propagation(nifti_path, mask_data, threshold_data, output_dir):
    """
    Process NIfTI file using SAM2 video propagation
    
    Args:
        nifti_path: Path to the NIfTI file
        mask_data: Mask data from draw step (3D numpy array)
        threshold_data: Threshold parameters
        output_dir: Output directory for results
    
    Returns:
        success: Boolean indicating success
        message: Status message
        results: Dictionary with processing results
    """
    try:
        # Load NIfTI file
        nii_img = nib.load(nifti_path)
        nii_data = nii_img.get_fdata()
        
        if len(nii_data.shape) != 3:
            return False, "NIfTI file must be 3D", None
        
        # Validate mask data
        if mask_data is None or mask_data.size == 0:
            return False, "Mask data is empty or None", None
        
        # Debug info
        print(f"NIfTI shape: {nii_data.shape}")
        print(f"Mask shape: {mask_data.shape}")
        print(f"Mask data type: {mask_data.dtype}")
        print(f"Mask non-zero count: {np.count_nonzero(mask_data) if mask_data.size > 0 else 0}")
        
        # Use safer ROI extraction
        roi_bounds, roi_slices, roi_masks, success = safe_roi_extraction(nii_data, mask_data)
        if not success:
            return False, "Failed to extract valid region of interest", None
        
        print(f"ROI bounds: {roi_bounds}")
        print(f"Number of ROI slices: {len(roi_slices)}")
        
        # Process slices for SAM2
        processed_slices = []
        
        for slice_idx, (roi_slice, roi_mask) in enumerate(zip(roi_slices, roi_masks)):
            if slice_idx == 0:
                # Apply threshold to first slice
                threshold_value = threshold_data.get('upper_threshold', 0.5)
                if isinstance(threshold_value, (int, float)):
                    thresholded = apply_safe_threshold(roi_slice, roi_mask, threshold_value)
                    processed_slices.append(thresholded)
                else:
                    processed_slices.append(safe_normalize_for_sam2(roi_slice))
            else:
                # Keep original for other slices
                processed_slices.append(safe_normalize_for_sam2(roi_slice))
        
        # Convert to format suitable for SAM2 video predictor
        video_frames = []
        for slice_data in processed_slices:
            # Convert to 3-channel RGB
            if len(slice_data.shape) == 2:
                slice_rgb = np.stack([slice_data] * 3, axis=-1)
            else:
                slice_rgb = slice_data
            
            video_frames.append(slice_rgb)
        
        # Validate that we have valid video frames
        if len(video_frames) == 0:
            return False, "No valid video frames generated", None
        
        # Check if frames have valid dimensions
        for i, frame in enumerate(video_frames):
            if frame.size == 0:
                return False, f"Empty frame at index {i}", None
        
        # Initialize SAM2 video manager
        sam2_video = SAM2VideoManager()
        
        # Load video model
        success, message = sam2_video.load_video_model()
        if not success:
            return False, f"Failed to load SAM2 video model: {message}", None
        
        # Initialize inference state
        success, message = sam2_video.init_inference_state(video_frames)
        if not success:
            return False, f"Failed to initialize inference state: {message}", None
        
        # Create initial mask from thresholded first frame using simple approach
        first_frame_thresholded = processed_slices[0]
        first_frame_mask = roi_masks[0]
        
        # Create simple initial mask
        initial_mask = create_simple_initial_mask(first_frame_thresholded, first_frame_mask)
        if initial_mask is None:
            return False, "Failed to create initial mask from thresholded data", None
        
        # Use SAM2 for automatic segmentation of the entire ROI (optional refinement)
        from sam_utils import SAM2Manager
        sam2_image = SAM2Manager()
        img_success, img_message = sam2_image.load_model()
        
        refined_mask = initial_mask.copy()  # Start with simple mask
        
        if img_success:
            # Use SAM2 to automatically segment using box prompt covering entire region
            set_success, set_message = sam2_image.set_image(video_frames[0])
            if set_success:
                try:
                    # Create a box covering the entire initial mask region
                    y_coords, x_coords = np.where(initial_mask > 0)
                    if len(x_coords) > 0 and len(y_coords) > 0:
                        try:
                            # Create bounding box covering the entire region
                            bbox_x1 = max(0, np.min(x_coords) - 5)
                            bbox_y1 = max(0, np.min(y_coords) - 5)
                            bbox_x2 = min(initial_mask.shape[1], np.max(x_coords) + 5)
                            bbox_y2 = min(initial_mask.shape[0], np.max(y_coords) + 5)
                            
                            # Use box prompt for automatic segmentation
                            input_box = np.array([bbox_x1, bbox_y1, bbox_x2, bbox_y2])
                            
                            masks, scores, pred_message = sam2_image.predict(
                                input_boxes=input_box[None, :],
                                multimask_output=False  # Single mask for the region
                            )
                            
                            if masks is not None and len(masks) > 0:
                                # Use SAM2 mask but constrain to original ROI
                                sam2_mask = masks[0].astype(np.uint8)
                                # Keep only parts within the original mask region
                                refined_mask = sam2_mask & (first_frame_mask > 0).astype(np.uint8)
                                
                                # Ensure we still have a valid mask
                                if np.sum(refined_mask) == 0:
                                    refined_mask = initial_mask  # Fallback
                        except Exception as inner_e:
                            print(f"Box coordinate calculation failed: {inner_e}")
                            refined_mask = initial_mask  # Fallback
                        
                except Exception as e:
                    print(f"SAM2 refinement failed: {e}")
                    refined_mask = initial_mask  # Fallback
        
        # Final validation of the mask to use
        if refined_mask.size == 0 or np.sum(refined_mask) == 0:
            return False, "Failed to generate valid segmentation mask for first frame", None
        
        # Add first frame mask to video predictor (use the refined mask)
        success, message = sam2_video.add_first_frame_mask(0, refined_mask)
        if not success:
            return False, f"Failed to add first frame mask: {message}", None
        
        # Propagate masks through all frames
        video_segments, prop_message = sam2_video.propagate_masks()
        if video_segments is None:
            return False, f"Failed to propagate masks: {prop_message}", None
        
        # Process results and save
        results = {
            'propagated_masks': video_segments,
            'original_slices': roi_slices,  # Use roi_slices instead of original_slices
            'roi_bounds': roi_bounds,
            'num_slices': len(processed_slices)
        }
        
        # Save propagated masks
        filename = os.path.splitext(os.path.basename(nifti_path))[0]
        mask_output_dir = os.path.join(output_dir, f"{filename}_sam2_masks")
        os.makedirs(mask_output_dir, exist_ok=True)
        
        # Save each propagated mask
        for frame_idx, frame_masks in video_segments.items():
            for obj_id, mask in frame_masks.items():
                mask_filename = f"slice_{frame_idx:03d}_obj_{obj_id}.png"
                mask_path = os.path.join(mask_output_dir, mask_filename)
                
                # Convert mask to image and save
                mask_img = (mask * 255).astype(np.uint8)
                Image.fromarray(mask_img).save(mask_path)
        
        # Create combined visualization
        create_sam2_visualization(
            roi_slices,  # Use roi_slices instead of original_slices
            video_segments, 
            roi_bounds,  # Use roi_bounds instead of mask_bounds
            os.path.join(output_dir, f"{filename}_sam2_visualization.png")
        )
        
        return True, "SAM2 propagation completed successfully", results
        
    except Exception as e:
        return False, f"Error processing with SAM2: {str(e)}", None

def create_sam2_visualization(original_slices, video_segments, mask_bounds, output_path):
    """Create visualization of SAM2 propagation results"""
    try:
        num_slices = len(original_slices)
        cols = min(8, num_slices)
        rows = (num_slices + cols - 1) // cols
        
        # Calculate dimensions for the combined image
        slice_size = 150  # Size for each slice in pixels
        fig_width = cols * slice_size
        fig_height = rows * slice_size
        
        # Create combined image
        combined_img = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)
        
        for i, slice_data in enumerate(original_slices):
            row = i // cols
            col = i % cols
            
            # Normalize slice
            slice_norm = safe_normalize_image(slice_data)
            
            # Resize slice to fit the grid
            slice_resized = cv2.resize(slice_norm, (slice_size, slice_size))
            
            # Get corresponding mask if available
            mask = None
            if i in video_segments and 1 in video_segments[i]:
                mask = video_segments[i][1]
                # Resize mask to match slice
                mask_resized = cv2.resize(mask.astype(np.uint8), (slice_size, slice_size))
            
            # Create overlay
            if mask is not None:
                # Create colored overlay
                overlay = np.stack([slice_resized] * 3, axis=-1)
                
                # Add mask in red with transparency
                mask_indices = mask_resized > 0.5
                overlay[mask_indices] = [255, slice_resized[mask_indices] // 2, slice_resized[mask_indices] // 2]
            else:
                overlay = np.stack([slice_resized] * 3, axis=-1)
            
            # Place in combined image
            y_start = row * slice_size
            y_end = y_start + slice_size
            x_start = col * slice_size
            x_end = x_start + slice_size
            
            combined_img[y_start:y_end, x_start:x_end] = overlay
        
        # Save combined visualization
        Image.fromarray(combined_img).save(output_path)
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")

def batch_sam2_process_step():
    """SAM2 batch processing step"""
    st.header("🤖 Step 4: SAM2 Processing")
    
    # Add consistent CSS styling
    st.markdown("""
    <style>
    .step-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2d3436;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(168, 237, 234, 0.3);
        border-left: 3px solid #00b894;
    }
    .sam2-info {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(116, 185, 255, 0.3);
    }
    .progress-section {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
        box-shadow: 0 1px 5px rgba(0,0,0,0.05);
    }
    .file-status {
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    .status-success {
        background: linear-gradient(135deg, #00b894 0%, #55efc4 100%);
        color: white;
    }
    .status-error {
        background: linear-gradient(135deg, #e17055 0%, #fdcb6e 100%);
        color: white;
    }
    .status-processing {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # SAM2 Mode Information
    st.markdown("""
    <div class="sam2-info">
        <h3>🤖 SAM2 Video Propagation Mode</h3>
        <p>This mode uses SAM2's video predictor for advanced segmentation propagation:</p>
        <ul>
            <li><strong>First Frame:</strong> Uses threshold + SAM2 inference for initial segmentation</li>
            <li><strong>Propagation:</strong> SAM2 automatically segments remaining slices based on the first frame</li>
            <li><strong>Consistency:</strong> Maintains temporal consistency across all slices</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if required data is available
    if "batch_files" not in st.session_state or "batch_final_thresholds" not in st.session_state:
        st.error("Missing required data. Please complete the previous steps.")
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        return
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check SAM2 availability
    sam2_manager = SAM2Manager()
    deps_ok, deps_msg = sam2_manager.check_dependencies()
    checkpoint_ok, checkpoint_msg = sam2_manager.check_checkpoint()
    
    if not deps_ok or not checkpoint_ok:
        st.error("SAM2 Setup Required")
        st.markdown(f"**Dependencies:** {deps_msg}")
        st.markdown(f"**Checkpoint:** {checkpoint_msg}")
        
        if not checkpoint_ok:
            st.info("""
            **Download SAM2 Checkpoint:**
            ```bash
            mkdir -p checkpoints
            wget -O checkpoints/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
            ```
            """)
        
        if st.button("← Back to Threshold Step"):
            st.session_state["current_step"] = "batch_threshold"
            st.rerun()
        return
    
    # Initialize processing state
    if "sam2_processing_started" not in st.session_state:
        st.session_state["sam2_processing_started"] = False
        st.session_state["sam2_completed_files"] = {}
        st.session_state["sam2_current_file_idx"] = 0
    
    files = st.session_state["batch_files"]
    thresholds = st.session_state["batch_final_thresholds"]
    
    # Progress overview
    st.markdown('<div class="progress-section">', unsafe_allow_html=True)
    st.markdown("### 📊 Processing Progress")
    
    total_files = len(files)
    completed_files = len(st.session_state["sam2_completed_files"])
    progress_percentage = (completed_files / total_files) * 100
    
    st.progress(progress_percentage / 100)
    st.markdown(f"**Progress:** {completed_files}/{total_files} files completed ({progress_percentage:.1f}%)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if not st.session_state["sam2_processing_started"]:
            if st.button("🚀 Start SAM2 Processing", type="primary", use_container_width=True):
                st.session_state["sam2_processing_started"] = True
                st.rerun()
    
    with col2:
        if completed_files > 0:
            if st.button("💾 Save Results", use_container_width=True):
                save_sam2_results()
                st.success("Results saved successfully!")
    
    with col3:
        if st.button("← Back to Threshold", use_container_width=True):
            st.session_state["current_step"] = "batch_threshold"
            st.rerun()
    
    # Process files
    if st.session_state["sam2_processing_started"]:
        current_idx = st.session_state["sam2_current_file_idx"]
        
        if current_idx < total_files:
            # Get file info - batch_files contains strings (filenames)
            filename = files[current_idx]
            filename_no_ext = filename.split('.')[0]
            
            # Build file path
            input_folder = os.path.join(os.getcwd(), 'media')
            file_path = os.path.join(input_folder, filename)
            
            st.markdown(f'<div class="file-status status-processing">🤖 Processing: {filename}</div>', unsafe_allow_html=True)
            
            # Get mask data from saved files (created in draw step)
            output_path = os.path.join(os.getcwd(), 'output', filename_no_ext)
            mask_path = os.path.join(output_path, 'dense.nii')
            
            if not os.path.exists(mask_path):
                st.error(f"Mask file not found: {mask_path}")
                st.session_state["sam2_completed_files"][filename] = {
                    "status": "error",
                    "message": "Mask file not found - draw step may not be completed"
                }
            else:
                # Load mask data from file
                try:
                    import nibabel as nib
                    mask_nii = nib.load(mask_path)
                    mask_data = mask_nii.get_fdata()
                    
                    # Debug mask data
                    st.write(f"Debug - Loaded mask shape: {mask_data.shape}")
                    st.write(f"Debug - Mask data type: {mask_data.dtype}")
                    st.write(f"Debug - Mask min/max: {np.min(mask_data)}/{np.max(mask_data)}")
                    st.write(f"Debug - Non-zero elements: {np.count_nonzero(mask_data)}")
                    
                    # Validate mask data
                    if mask_data.size == 0:
                        st.error(f"Loaded mask is empty for {filename}")
                        st.session_state["sam2_completed_files"][filename] = {
                            "status": "error",
                            "message": "Loaded mask is empty"
                        }
                        continue
                    
                    if np.count_nonzero(mask_data) == 0:
                        st.error(f"Loaded mask contains no non-zero values for {filename}")
                        st.session_state["sam2_completed_files"][filename] = {
                            "status": "error",
                            "message": "Mask contains no non-zero values"
                        }
                        continue
                    
                    # Get threshold data from session state
                    threshold_data = thresholds.get(filename_no_ext, {})
                    
                    if not threshold_data:
                        st.error(f"No threshold data found for {filename}")
                        st.session_state["sam2_completed_files"][filename] = {
                            "status": "error", 
                            "message": "No threshold data found"
                        }
                    else:
                        # Convert threshold data to proper format if needed
                        if isinstance(threshold_data, (int, float)):
                            # If threshold_data is just a number, convert to dict format
                            threshold_dict = {
                                'lower_threshold': 0.0,
                                'upper_threshold': float(threshold_data)
                            }
                        else:
                            threshold_dict = threshold_data
                        
                        # Process with SAM2
                        success, message, results = process_nifti_with_sam2_propagation(
                            file_path, mask_data, threshold_dict, output_dir
                        )
                        
                        st.session_state["sam2_completed_files"][filename] = {
                            "status": "success" if success else "error",
                            "message": message,
                            "results": results
                        }
                        
                        if success:
                            st.success(f"✅ {filename}: {message}")
                        else:
                            st.error(f"❌ {filename}: {message}")
                            
                except Exception as e:
                    st.error(f"Error loading mask data: {str(e)}")
                    st.session_state["sam2_completed_files"][filename] = {
                        "status": "error",
                        "message": f"Error loading mask data: {str(e)}"
                    }
            
            # Move to next file
            st.session_state["sam2_current_file_idx"] += 1
            st.rerun()
        
        else:
            # All files processed
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            st.markdown("### 🎉 SAM2 Processing Complete!")
            st.markdown(f"Successfully processed {total_files} files using SAM2 video propagation.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show completion statistics
            success_count = sum(1 for result in st.session_state["sam2_completed_files"].values() 
                              if result["status"] == "success")
            error_count = total_files - success_count
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("✅ Successful", success_count)
            with col2:
                st.metric("❌ Errors", error_count)
    
    # Show detailed results
    if st.session_state["sam2_completed_files"]:
        st.markdown('<div class="progress-section">', unsafe_allow_html=True)
        st.markdown("### 📋 Detailed Results")
        
        for filename, result in st.session_state["sam2_completed_files"].items():
            status_class = "status-success" if result["status"] == "success" else "status-error"
            status_icon = "✅" if result["status"] == "success" else "❌"
            
            st.markdown(f'<div class="file-status {status_class}">{status_icon} {filename}: {result["message"]}</div>', 
                       unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def save_sam2_results():
    """Save SAM2 processing results"""
    try:
        output_dir = "output"
        results_file = os.path.join(output_dir, "sam2_processing_results.json")
        
        # Prepare results for JSON serialization
        json_results = {}
        for filename, result in st.session_state["sam2_completed_files"].items():
            json_results[filename] = {
                "status": result["status"],
                "message": result["message"],
                "has_results": result.get("results") is not None
            }
        
        with open(results_file, 'w') as f:
            import json
            json.dump(json_results, f, indent=2)
        
        st.success(f"Results saved to {results_file}")
        
    except Exception as e:
        st.error(f"Error saving results: {str(e)}")
