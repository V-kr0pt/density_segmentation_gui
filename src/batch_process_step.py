# =========================
# Imports
# =========================
import os
import time
import threading
import traceback
import gc
import numpy as np
import logging
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from ImageLoader import UnifiedImageLoader
from new_utils import MaskManager, DisplayTransform, resolve_dense_mask_path
from performance_config import performance_config, get_system_info

# GPU acceleration with automatic CPU fallback
try:
    from gpu_accelerator import (
        GPUThresholdOperator,
        GPUBatchProcessor,
        GPUMaskOperator,
        gpu_config
    )
    GPU_AVAILABLE = gpu_config.cuda_available
except ImportError:
    from new_utils import ThresholdOperator as GPUThresholdOperator
    from new_utils import MaskManager as GPUMaskOperator
    GPU_AVAILABLE = False
    GPUBatchProcessor = None

logger = logging.getLogger(__name__)


# =========================
# Multi-threading Manager Class
# =========================
class BatchProcessingManager:
    """
    Manages multi-threaded batch file processing with GPU acceleration.
    """
    
    def __init__(self, max_workers=None):
        self.max_workers = max_workers or performance_config.get_optimal_workers()
        self.progress_lock = threading.Lock()
        self.error_lock = threading.Lock()
        self.memory_lock = threading.Lock()
        self.operations_count = 0
        self.performance_settings = performance_config.get_memory_settings()
        self.io_settings = performance_config.get_io_settings()
        
        # GPU configuration
        self.use_gpu = GPU_AVAILABLE
        if self.use_gpu and GPUBatchProcessor is not None:
            self.gpu_processor = GPUBatchProcessor()
            logger.info("GPU acceleration enabled for batch processing")
        else:
            self.gpu_processor = None
        
    def process_single_file(self, file_info):
        """
        Process a single file in a thread-safe manner.
        """
        try:
            file = file_info['file']
            file_name = file_info['file_name']
            input_folder = file_info['input_folder']
            final_thresholds = file_info['final_thresholds']
            
            # File paths setup
            original_image_path = os.path.join(input_folder, file)
            output_path = os.path.join(os.getcwd(), 'output', file_name)
            is_dicom = os.path.isdir(original_image_path) or original_image_path.lower().endswith((".dcm", ".dicom"))
            mask_path = resolve_dense_mask_path(output_path, prefer_dicom=is_dicom)
            save_dir = os.path.join(output_path, 'dense_mask')
            
            # Get threshold value
            T = final_thresholds[file_name]
            
            # Validate file existence
            if not os.path.exists(original_image_path):
                raise FileNotFoundError(f"Original image not found: {original_image_path}")
            if mask_path is None:
                raise FileNotFoundError("Mask not found for this case.")
            
            # Load central slice to calculate target area
            central_image_slice, original_affine, original_shape, central_slice_idx, img_type = \
                UnifiedImageLoader.load_slice(original_image_path)
            central_mask_slice, _, _, _, _ = UnifiedImageLoader.load_slice(mask_path, central_slice_idx)
            
            # Calculate target area from central slice (use GPU if available)
            if self.use_gpu:
                thresholded_img = GPUThresholdOperator.threshold_slice(
                    central_image_slice, central_mask_slice, T
                )
                target_area = GPUMaskOperator.measure_mask_area(thresholded_img)
            else:
                from new_utils import ThresholdOperator
                thresholded_img = ThresholdOperator.threshold_slice(
                    central_image_slice, central_mask_slice, T
                )
                target_area = MaskManager.measure_mask_area(thresholded_img)
            
            logger.debug(f"Processing {file_name}: shape={original_shape}, "
                        f"slices={original_shape[np.argmin(original_shape)]}, "
                        f"target_area={target_area}")
            
            # Clean output directory
            if os.path.exists(save_dir):
                for f in os.listdir(save_dir):
                    os.remove(os.path.join(save_dir, f))
            os.makedirs(save_dir, exist_ok=True)
            
            # Determine number of slices
            slice_dim = np.argmin(original_shape)
            num_slices = original_shape[slice_dim]
            
            # Process all slices with GPU acceleration if available
            slice_results = self._process_slices_optimized(
                original_image_path, mask_path, save_dir,
                target_area, num_slices, img_type
            )
            
            # Create final NIfTI file from processed slices
            nifti_path = MaskManager.create_final_mask(
                save_dir,
                original_shape,
                original_affine,
                img_type=img_type,
                original_image_path=original_image_path,
            )
            
            return {
                'success': True,
                'file_name': file_name,
                'file': file,
                'nifti_path': nifti_path,
                'slices_processed': slice_results
            }
            
        except Exception as e:
            error_msg = f"Error processing {file}: {str(e)}\n{traceback.format_exc()}"
            with self.error_lock:
                logger.error(error_msg)
            return {
                'success': False,
                'file_name': file_name,
                'file': file,
                'error': error_msg
            }
    
    def _process_slices_optimized(self, original_image_path, mask_path, save_dir,
                                 target_area, num_slices, img_type):
        """
        Process all slices with GPU acceleration or CPU fallback.
        Uses GPU batch processing for significant speedups when available.
        """
        # Use GPU batch processing for larger datasets
        if self.use_gpu and self.gpu_processor is not None and num_slices > 10:
            return self._process_slices_gpu_batch(
                original_image_path, mask_path, save_dir,
                target_area, num_slices, img_type
            )
        else:
            return self._process_slices_cpu(
                original_image_path, mask_path, save_dir,
                target_area, num_slices, img_type
            )
    
    def _process_slices_gpu_batch(self, original_image_path, mask_path, save_dir,
                                  target_area, num_slices, img_type):
        """GPU-accelerated batch slice processing."""
        # Load all slices
        image_slices = []
        mask_slices = []
        
        for slice_idx in range(num_slices):
            img_slice, _, _, _, _ = UnifiedImageLoader.load_slice(
                original_image_path, slice_idx
            )
            msk_slice, _, _, _, _ = UnifiedImageLoader.load_slice(
                mask_path, slice_idx
            )
            image_slices.append(img_slice)
            mask_slices.append(msk_slice)
        
        # Process all slices on GPU
        results = self.gpu_processor.process_slices_batch(
            image_slices, mask_slices, target_area
        )
        
        # Save results
        display_transform = DisplayTransform(padding=0)
        display_transform.set_rotation_for_type(img_type)
        compression_level = self.io_settings['compression_level']
        
        for slice_idx, (threshold, binary_image) in enumerate(results):
            # Save numpy array
            npy_filename = f'slice_{slice_idx:04d}_threshold_{threshold:.2f}.npy'
            np.save(os.path.join(save_dir, npy_filename), binary_image)
            
            # Save PNG for visualization
            png_filename = f'slice_{slice_idx:04d}_threshold_{threshold:.2f}.png'
            if display_transform.rotate_k % 4 != 0:
                binary_img_display = np.rot90(binary_image, k=display_transform.rotate_k)
            else:
                binary_img_display = binary_image
            
            img_pil = Image.fromarray(binary_img_display, mode='L')
            img_pil.save(os.path.join(save_dir, png_filename), optimize=True, compress_level=compression_level)
        
        return len(results)
    
    def _process_slices_cpu(self, original_image_path, mask_path, save_dir,
                           target_area, num_slices, img_type):
        """
        CPU fallback: Process slices with chunking and memory management.
        """
        from new_utils import ThresholdOperator
        
        chunk_size = performance_config.get_chunk_size(num_slices)
        total_processed = 0
        
        compression_level = self.io_settings['compression_level']
        gc_frequency = self.performance_settings['gc_frequency']
        
        display_transform = DisplayTransform(padding=0)
        display_transform.set_rotation_for_type(img_type)

        for start_idx in range(0, num_slices, chunk_size):
            end_idx = min(start_idx + chunk_size, num_slices)
            
            for slice_index in range(start_idx, end_idx):
                try:
                    # Load slice in NATIVE orientation
                    image_slice, _, _, _, _ = UnifiedImageLoader.load_slice(
                        original_image_path, slice_index
                    )
                    mask_slice, _, _, _, _ = UnifiedImageLoader.load_slice(
                        mask_path, slice_index
                    )
                    
                    # Apply dynamic threshold adjustment
                    adjusted_threshold, thresholded_image = \
                        ThresholdOperator.adjust_slice_threshold(
                            image_slice, mask_slice, target_area
                        )
                    
                    # Convert to binary (0 or 255 for visualization)
                    binary_image = np.where(thresholded_image > 0, 255, 0).astype(np.uint8)
                    
                    # Save as numpy array (in NATIVE orientation)
                    npy_filename = f'slice_{slice_index:04d}_threshold_{adjusted_threshold:.2f}.npy'
                    npy_filepath = os.path.join(save_dir, npy_filename)
                    np.save(npy_filepath, binary_image)
                    
                    # Also save PNG for visualization (rotated for display)
                    png_filename = f'slice_{slice_index:04d}_threshold_{adjusted_threshold:.2f}.png'
                    png_filepath = os.path.join(save_dir, png_filename)
                    
                    # Apply rotation for PNG visualization (match GUI display)
                    if display_transform.rotate_k % 4 != 0:
                        binary_img_display = np.rot90(binary_image, k=display_transform.rotate_k)
                    else:
                        binary_img_display = binary_image
                    img_pil = Image.fromarray(binary_img_display, mode='L')
                    img_pil.save(png_filepath, optimize=True, compress_level=compression_level)
                    
                    total_processed += 1
                    
                    # Memory management
                    with self.memory_lock:
                        self.operations_count += 1
                        if self.operations_count % gc_frequency == 0:
                            gc.collect()
                    
                    # Clean up large variables
                    del image_slice, mask_slice, thresholded_image, binary_image
                    
                except Exception as e:
                    logger.warning(f"Error processing slice {slice_index}: {str(e)}")
                    continue
        
        return total_processed
    
    def process_files_batch(self, files_to_process, input_folder, final_thresholds,
                           progress_callback=None):
        """
        Process multiple files using ThreadPoolExecutor.
        """
        results = {
            'completed': [],
            'errors': [],
            'total_time': 0
        }
        
        start_time = time.time()
        
        # Prepare file information
        file_infos = []
        for file in files_to_process:
            file_name = file.split('.')[0]  # Remove extension
            file_infos.append({
                'file': file,
                'file_name': file_name,
                'input_folder': input_folder,
                'final_thresholds': final_thresholds
            })
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_single_file, info): info
                for info in file_infos
            }
            
            completed_count = 0
            for future in as_completed(future_to_file):
                file_info = future_to_file[future]
                try:
                    result = future.result()
                    
                    if result['success']:
                        results['completed'].append(result)
                    else:
                        results['errors'].append(result)
                    
                    completed_count += 1
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(
                            completed_count,
                            len(files_to_process),
                            file_info['file']
                        )
                        
                except Exception as e:
                    results['errors'].append({
                        'file': file_info['file'],
                        'file_name': file_info['file_name'],
                        'error': str(e)
                    })
        
        results['total_time'] = time.time() - start_time
        return results


# =========================
# Main Batch Process Step Function
# =========================
def batch_process_step():
    """
    Step 4: Batch processing with multi-threading support.
    """
    
    # =========================
    # UI Header & Styling
    # =========================
    st.header("⚙️ Step 4: Process Files")
    with open("static/batch_process_step.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # =========================
    # Session State Validation
    # =========================
    if "batch_files" not in st.session_state:
        st.error("No batch files selected. Please go back to file selection.")
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        return
    
    if "batch_final_thresholds" not in st.session_state:
        st.error("No thresholds set. Please complete the threshold step first.")
        if st.button("← Back to Threshold Step"):
            st.session_state["current_step"] = "batch_threshold"
            st.rerun()
        return
    
    # =========================
    # Prepare Data
    # =========================
    batch_files = st.session_state["batch_files"]
    final_thresholds = st.session_state["batch_final_thresholds"]
    input_folder = st.session_state.get("main_input_folder", os.path.join(os.getcwd(), "media"))
    
    completed_threshold = st.session_state["batch_completed_files"]["threshold"]
    completed_process = st.session_state["batch_completed_files"]["process"]
    batch_files_without_extension = [f.split('.')[0] for f in batch_files]
    completed_process_filtered = [f for f in completed_process if f in batch_files_without_extension]
    
    # Filter files that have thresholds but not yet processed
    files_to_process = []
    for file in batch_files:
        file_name = file.split('.')[0]
        if file_name in completed_threshold and file_name not in completed_process:
            files_to_process.append(file)
    
    # =========================
    # Progress Display
    # =========================
    total_files = len(batch_files)
    st.write(f"### Progress: {len(completed_process_filtered)}/{total_files} files processed")
    st.progress(len(completed_process_filtered) / total_files)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # =========================
    # All Files Processed Message
    # =========================
    if len(files_to_process) == 0 and len(completed_process_filtered) == total_files:
        st.markdown("""
        <div class="success-container">
            <h4>🎉 All files have been processed successfully!</h4>
            <p>All segmentation masks have been generated and saved to the output directory.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 Start New Batch"):
            keys_to_remove = [key for key in st.session_state.keys() if key.startswith("batch_")]
            for key in keys_to_remove:
                del st.session_state[key]
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        return
    
    # =========================
    # No Files Pending Message
    # =========================
    if len(files_to_process) == 0:
        st.info("No files pending for processing.")
        if st.button("← Back to Threshold Step"):
            st.session_state["current_step"] = "batch_threshold"
            st.rerun()
        return
    
    # =========================
    # Batch Processing Section
    # =========================
    st.write(f"### Ready to process {len(files_to_process)} files")
    
    # System information
    with st.expander("📊 System Information & Performance Settings"):
        system_info = get_system_info()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU Cores", system_info['cpu_count'])
            st.metric("Available Memory", f"{system_info['memory_available_mb']} MB")
        
        with col2:
            if 'cpu_percent' in system_info:
                st.metric("Current CPU Usage", f"{system_info['cpu_percent']:.1f}%")
    
    # Thread configuration
    default_workers = performance_config.get_optimal_workers(len(files_to_process))
    
    max_workers = st.number_input(
        "Number of parallel threads:",
        min_value=1,
        max_value=128,
        value=default_workers,
        step=1,
        help=f"Recommended: {default_workers} threads based on system resources"
    )
    
    if st.button("🚀 Process All Files (Multi-threaded)", type="primary"):
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        def update_progress(completed, total, current_file):
            progress_placeholder.progress(
                completed / total,
                text=f"Processing: {completed}/{total} files completed"
            )
            
            elapsed_time = time.time() - start_time
            if completed > 0:
                avg_time = elapsed_time / completed
                remaining_time = avg_time * (total - completed)
                status_placeholder.text(
                    f"✅ Latest: {current_file} | "
                    f"Est. remaining: {remaining_time:.1f}s"
                )
        
        # Execute processing
        batch_manager = BatchProcessingManager(max_workers=max_workers)
        
        try:
            start_time = time.time()
            results = batch_manager.process_files_batch(
                files_to_process,
                input_folder,
                final_thresholds,
                update_progress
            )
            
            # Update session state
            for result in results['completed']:
                if result['file_name'] not in st.session_state["batch_completed_files"]["process"]:
                    st.session_state["batch_completed_files"]["process"].append(result['file_name'])
            
            # Show results
            if results['completed']:
                st.success(f"✅ {len(results['completed'])} files processed successfully!")
                st.info(f"⏱️ Total time: {results['total_time']:.2f}s | "
                       f"🚀 Speed: {len(results['completed'])/results['total_time']:.2f} files/s | "
                       f"🧵 Threads: {max_workers}")
            
            if results['errors']:
                st.error(f"❌ {len(results['errors'])} files failed:")
                for error in results['errors']:
                    with st.expander(f"Error details for {error['file']}"):
                        st.code(error['error'])
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Error during parallel processing: {str(e)}")
            st.exception(e)
    
    # =========================
    # Individual File Processing
    # =========================
    st.divider()
    st.markdown("""
    <div class="step-container">
        <h4>🔧 Individual Processing</h4>
        <p>Process files one by one if you prefer more control.</p>
    </div>
    """, unsafe_allow_html=True)
    
    def strip_known_extension(filename):
        known_exts = ('.nii.gz', '.nii', '.dcm', '.dicom')
        for ext in known_exts:
            if filename.lower().endswith(ext):
                return filename[:-len(ext)]
        return filename
    
    for file in files_to_process:
        file_name = strip_known_extension(file)
        threshold = final_thresholds[file_name]
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 5px; margin: 0.25rem 0; border-left: 3px solid #007bff;">
                <strong>{file}</strong><br>
                Threshold: <code>{threshold:.3f}</code>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("Process", key=f"process_{file_name}"):
                batch_manager = BatchProcessingManager(max_workers=1)
                
                file_info = {
                    'file': file,
                    'file_name': file_name,
                    'input_folder': input_folder,
                    'final_thresholds': final_thresholds
                }
                
                try:
                    with st.spinner(f"Processing {file}..."):
                        start_time = time.time()
                        result = batch_manager.process_single_file(file_info)
                        end_time = time.time()
                        
                        if result['success']:
                            st.session_state["batch_completed_files"]["process"].append(file_name)
                            st.success(f"✅ {file} processed successfully! "
                                     f"({end_time - start_time:.2f}s)")
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {result['error']}")
                            
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    st.exception(e)
    
    # =========================
    # Back Button
    # =========================
    if st.button("← Back to Threshold Step"):
        st.session_state["current_step"] = "batch_threshold"
        st.rerun()
