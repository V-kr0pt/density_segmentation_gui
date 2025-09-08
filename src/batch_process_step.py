# =========================
# Imports
# =========================
import os
import time
import threading
import traceback
import gc
import numpy as np
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from utils import ImageOperations, MaskOperations, ThresholdOperations
from performance_config import performance_config, get_system_info


# =========================
# Multi-threading Manager Class
# =========================
class BatchProcessingManager:
    """
    Manages multi-threaded batch file processing.
    Implements Python threading best practices for optimized performance.
    """
    
    def __init__(self, max_workers=None):
        """
        Initialize the batch processing manager.
        
        Args:
            max_workers (int): Maximum number of threads. If None, uses optimized configuration
        """
        self.max_workers = max_workers or performance_config.get_optimal_workers()
        self.progress_lock = threading.Lock()
        self.error_lock = threading.Lock()
        self.memory_lock = threading.Lock()
        self.operations_count = 0
        self.performance_settings = performance_config.get_memory_settings()
        self.io_settings = performance_config.get_io_settings()
        
    def process_single_file(self, file_info):
        """
        Process a single file in a thread-safe manner.
        
        Args:
            file_info (dict): File information to be processed
            
        Returns:
            dict: Processing result
        """
        try:
            file = file_info['file']
            file_name = file_info['file_name']
            input_folder = file_info['input_folder']
            final_thresholds = file_info['final_thresholds']
            
            # File paths setup
            original_image_path = os.path.join(input_folder, file)
            output_path = os.path.join(os.getcwd(), 'output', file_name)
            mask_path = os.path.join(output_path, 'dense.nii')
            save_dir = os.path.join(output_path, 'dense_mask')
            
            # Get threshold value
            T = final_thresholds[file_name]
            
            # Validate file existence
            if not os.path.exists(original_image_path):
                raise FileNotFoundError(f"Original image not found: {original_image_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            
            # Load data and calculate target area
            _, original_affine, num_slices = ImageOperations.load_image(original_image_path)
            middle_slice_index = num_slices // 2
            middle_image_slice = ImageOperations.load_any_slice(original_image_path, middle_slice_index)
            middle_mask_slice = ImageOperations.load_nii_slice(mask_path, middle_slice_index)
            thresholded_img = ThresholdOperations.threshold_image(middle_image_slice, middle_mask_slice, T)
            target_area = MaskOperations.measure_mask_area(thresholded_img)
            
            # Clean output directory
            if os.path.exists(save_dir):
                for f in os.listdir(save_dir):
                    os.remove(os.path.join(save_dir, f))
            os.makedirs(save_dir, exist_ok=True)
            
            # Process all slices with optimized chunking
            slice_results = self._process_slices_optimized(
                original_image_path, mask_path, save_dir, 
                target_area, num_slices, file_name
            )
            
            # Create NIfTI file
            nifti_path = MaskOperations.create_mask_nifti(save_dir, original_affine)
            
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
                pass  # Could log errors here if needed
            return {
                'success': False,
                'file_name': file_name,
                'file': file,
                'error': error_msg
            }
    
    def _process_slices_optimized(self, original_image_path, mask_path, save_dir, 
                                 target_area, num_slices, file_name):
        """
        Process slices with optimized chunking and memory management.
        """
        # Chunk size based on number of slices
        chunk_size = performance_config.get_chunk_size(num_slices)
        total_processed = 0
        
        # Optimized I/O settings
        compression_level = self.io_settings['compression_level']
        gc_frequency = self.performance_settings['gc_frequency']
        
        for start_idx in range(0, num_slices, chunk_size):
            end_idx = min(start_idx + chunk_size, num_slices)
            
            # Process current chunk
            for slice_index in range(start_idx, end_idx):
                try:
                    # Load slice data with lazy loading
                    image_slice = ImageOperations.load_any_slice(original_image_path, slice_index)
                    mask_slice = ImageOperations.load_nii_slice(mask_path, slice_index)
                    mask_slice = np.flip(mask_slice, axis=1)
                    
                    # Apply dynamic threshold adjustment
                    adjusted_threshold, thresholded_image = ThresholdOperations.adjust_threshold(
                        image_slice, mask_slice, target_area, slice_index
                    )
                    
                    # Create binary image and save
                    binary_image = np.where(thresholded_image > 0, 255, 0).astype(np.uint8)
                    filename = f'slice_{slice_index}_threshold_{adjusted_threshold:.4f}.png'
                    filepath = os.path.join(save_dir, filename)
                    
                    # Optimized saving with compression
                    img_pil = Image.fromarray(binary_image.T, mode='L')
                    img_pil.save(filepath, optimize=True, compress_level=compression_level)
                    
                    total_processed += 1
                    
                    # Memory management
                    with self.memory_lock:
                        self.operations_count += 1
                        if self.operations_count % gc_frequency == 0:
                            gc.collect()
                    
                    # Clean up large variables
                    del image_slice, mask_slice, thresholded_image, binary_image
                    
                except Exception as e:
                    # Log slice-specific error
                    continue
        
        return total_processed
    
    def process_files_batch(self, files_to_process, input_folder, final_thresholds, 
                           progress_callback=None):
        """
        Process multiple files using ThreadPoolExecutor.
        
        Args:
            files_to_process (list): List of files to process
            input_folder (str): Input folder path
            final_thresholds (dict): Dictionary with final thresholds
            progress_callback (callable): Callback for progress updates
            
        Returns:
            dict: Processing results
        """
        results = {
            'completed': [],
            'errors': [],
            'total_files': len(files_to_process),
            'total_time': 0
        }
        
        start_time = time.time()
        
        # Prepare file information
        file_infos = []
        for file in files_to_process:
            file_name = file.split('.')[0] if file.endswith('.nii') or file.endswith('.nii.gz') else file
            file_infos.append({
                'file': file,
                'file_name': file_name,
                'input_folder': input_folder,
                'final_thresholds': final_thresholds
            })
        
        # Execute parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_single_file, file_info): file_info['file']
                for file_info in file_infos
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    if result['success']:
                        results['completed'].append(result)
                        if progress_callback:
                            progress_callback(len(results['completed']), len(files_to_process), file)
                    else:
                        results['errors'].append(result)
                except Exception as e:
                    error_msg = f"Unexpected error processing {file}: {str(e)}"
                    results['errors'].append({
                        'file': file,
                        'error': error_msg,
                        'success': False
                    })
        
        results['total_time'] = time.time() - start_time
        return results



def batch_process_step():
    """
    Main function for Step 4: Process Files in batch mode.
    Handles batch and individual file processing, progress, and navigation.
    """

    # =========================
    # UI Header & Styling
    # =========================
    st.header("Step 4: Process Files")
    with open("static/batch_process_step.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # =========================
    # Input Folder Setup
    # =========================
    input_folder = os.path.join(os.getcwd(), 'media')

    # =========================
    # Batch Data Validation
    # =========================
    if "batch_files" not in st.session_state:
        st.error("No batch files selected. Please go back to file selection.")
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        return

    # =========================
    # Batch State Setup
    # =========================
    batch_files = st.session_state["batch_files"]
    completed_draw = st.session_state["batch_completed_files"]["draw"]
    completed_threshold = st.session_state["batch_completed_files"]["threshold"]
    all_completed_completed_process = st.session_state["batch_completed_files"]["process"]
    completed_process = [f for f in all_completed_completed_process if f in st.session_state["batch_files_without_extension"]]

    # =========================
    # Prerequisite Checks
    # =========================
    if len(completed_draw) < len(batch_files):
        st.warning(f"Please complete all drawing steps first. {len(completed_draw)}/{len(batch_files)} completed.")
        if st.button("← Back to Draw Step"):
            st.session_state["current_step"] = "batch_draw"
            st.rerun()
        return

    if len(completed_threshold) < len(batch_files):
        st.warning(f"Please complete all threshold steps first. {len(completed_threshold)}/{len(batch_files)} completed.")
        if st.button("← Back to Threshold Step"):
            st.session_state["current_step"] = "batch_threshold"
            st.rerun()
        return

    if "batch_final_thresholds" not in st.session_state:
        st.error("No thresholds found. Please complete the threshold step first.")
        if st.button("← Back to Threshold Step"):
            st.session_state["current_step"] = "batch_threshold"
            st.rerun()
        return

    total_files = len(batch_files)

    # =========================
    # Progress Overview
    # =========================
    st.write(f"### Progress: {len(completed_process)}/{total_files} files completed")
    st.progress(len(completed_process) / total_files)
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # Files Status Display
    # =========================
    st.write("### Files Status")
    final_thresholds = st.session_state["batch_final_thresholds"]
    files_to_process = []
    for file in batch_files:
        file_name = file.split('.')[0]
        if file_name in completed_draw and file_name in completed_threshold:
            threshold = final_thresholds.get(file_name, "Not set")
            if file_name in completed_process:
                status = "✅ Completed"
                color = "#28a745"
            else:
                status = "⏳ Ready"
                color = "#ffc107"
                files_to_process.append(file)
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 5px; margin: 0.25rem 0; border-left: 3px solid {color};">
                <strong>{file}</strong><br>
                Threshold: <code>{threshold:.3f}</code> | Status: <span style="color: {color};">{status}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 0.75rem; border-radius: 5px; margin: 0.25rem 0; border-left: 3px solid #dc3545;">
                <strong>{file}</strong><br>
                <span style="color: #dc3545;">❌ Missing prerequisites</span>
            </div>
            """, unsafe_allow_html=True)

    # =========================
    # All Files Processed Message
    # =========================
    if len(files_to_process) == 0 and len(completed_process) == total_files:
        st.markdown("""
        <div class="success-container">
            <h4>🎉 All files have been processed successfully!</h4>
            <p>All segmentation masks have been generated and saved to the output directory.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 Start New Batch"):
            # Clear all batch-related session state
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
    
    # System information and settings
    with st.expander("📊 System Information & Performance Settings"):
        system_info = get_system_info()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU Cores", system_info['cpu_count'])
            st.metric("Available Memory", f"{system_info['memory_available_mb']} MB")
        
        with col2:
            if 'cpu_percent' in system_info:
                st.metric("Current CPU Usage", f"{system_info['cpu_percent']:.1f}%")
    
    # Processing configuration
    default_workers = performance_config.get_optimal_workers(len(files_to_process))
    
    # Calculate explanation for tooltip
    cpu_factor = f"CPU cores ({system_info['cpu_count']}) × 2"
    memory_factor = f"Memory ({system_info['memory_available_mb']}MB ÷ 500MB per thread)"
    workload_factor = f"Workload (min of {len(files_to_process)} files, max 6)"
    
    # Manual thread selection
    max_workers = st.number_input(
        "Number of parallel threads:",
        min_value=1,
        max_value=128,
        value=default_workers,
        step=1,
        help=f"Recommended: {default_workers} threads\n\n"
             f"Calculation based on:\n"
             f"• {cpu_factor}\n"
             f"• {memory_factor}\n" 
             f"• {workload_factor}\n\n"
             f"The system automatically chooses the smallest value to prevent resource exhaustion."
    )
    
    if st.button("🚀 Process All Files (Multi-threaded)", type="primary"):
        # Initialize progress tracking
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Update the existing progress section
        def update_progress(completed, total, current_file):
            # Update the main progress bar that already exists
            progress_placeholder.progress(
                completed / total, 
                text=f"Processing: {completed}/{total} files completed"
            )
            
            # Show current file status
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
            
            # Update session state with completed files
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
    # Individual File Processing Section
    # =========================
    st.divider()
    st.markdown("""
    <div class="step-container">
        <h4>🔧 Individual Processing</h4>
        <p>Process files one by one if you prefer more control over each file.</p>
    </div>
    """, unsafe_allow_html=True)

    for file in files_to_process:
        file_name = file.split('.')[0] if file.endswith('.nii') or file.endswith('.nii.gz') else file            
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
                # Use optimized manager for individual processing
                batch_manager = BatchProcessingManager(max_workers=1)  # Single thread for individual
                
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
                    st.error(f"❌ Unexpected error processing {file}: {str(e)}")
                    st.exception(e)

    # =========================
    # Back Button
    # =========================
    if st.button("← Back to Threshold Step"):
        st.session_state["current_step"] = "batch_threshold"
        st.rerun()
