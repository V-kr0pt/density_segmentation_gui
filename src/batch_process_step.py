# =========================
# Imports
# =========================
import os
import sys
import time
import json
import tempfile
import subprocess
import threading
import traceback
import gc
import numpy as np
import nibabel as nib
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from ImageLoader import UnifiedImageLoader
from new_utils import (
    ThresholdOperator,
    MaskManager,
    DisplayTransform,
    resolve_dense_mask_path,
    sagittal_cw_rotate,
    apply_orientation,
    orient_shape_3d,
    load_polygon_mask_2d,
    has_polygon_mask,
)
from performance_config import performance_config, get_system_info


def _read_orientation_json(output_path: str) -> dict:
    """
    Load orientation saved by the draw step. Falls back to identity if the
    file is missing or unreadable (older runs without orientation support).
    """
    path = os.path.join(output_path, "orientation.json")
    if os.path.isfile(path):
        try:
            with open(path) as f:
                data = json.load(f)
            return {
                "rotate_k": int(data.get("rotate_k", 0)) % 4,
                "flip_h": bool(data.get("flip_h", False)),
                "flip_v": bool(data.get("flip_v", False)),
            }
        except Exception:
            pass
    return {"rotate_k": 0, "flip_h": False, "flip_v": False}


# =========================
# Hard resource caps for background processing
# Chosen empirically: two concurrent DBT files peak around 3–3.5 GB, leaving
# headroom for the Streamlit UI process. Raising these has caused OOM.
# =========================
_MAX_BG_WORKERS = 2
_MAX_BG_MEMORY_BYTES = 4 * 1024 ** 3  # 4 GiB across all active workers


def _workers_memory_bytes(active) -> int:
    """Sum RSS (resident set size) across all currently-running worker processes."""
    try:
        import psutil
    except ImportError:
        return 0
    total = 0
    for item in active:
        try:
            p = psutil.Process(item["proc"].pid)
            total += p.memory_info().rss
            for child in p.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            continue
    return total


# =========================
# Background job state
# Module-level so it persists across Streamlit reruns
# =========================
_BG_JOB: dict = {
    "running":    False,
    "completed":  [],
    "errors":     [],
    "total":      0,
    "done":       0,
    "start_time": None,
    "end_time":   None,
}
_BG_LOCK = threading.Lock()


def _run_bg_processing(files_to_process, input_folder, final_thresholds, max_workers):
    """
    Runs in a lightweight daemon thread; dispatches each file to a separate
    subprocess worker (_process_worker.py) and updates _BG_JOB as results
    come in.

    Why subprocesses, not threads:
      pydicom decompression and the per-slice threshold loop hold the GIL
      enough to starve Streamlit's UI thread when several workers run in the
      same process. Subprocess workers each own their GIL, so they can't
      block the Streamlit main thread regardless of what numpy/pydicom do
      internally. This daemon itself only polls Popen handles and touches
      temp files — no numpy/pydicom work here.
    """
    global _BG_JOB

    # Hard-cap workers regardless of what the caller requested — two
    # concurrent DBT decompressions already push the process near the
    # memory budget.
    max_workers = max(1, min(int(max_workers), _MAX_BG_WORKERS))

    worker_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_process_worker.py"
    )
    creationflags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW

    pending = [
        {
            'file':             f,
            'file_name':        f.split('.')[0],
            'input_folder':     input_folder,
            'final_thresholds': final_thresholds,
        }
        for f in files_to_process
    ]
    active = []  # list of dicts: {proc, info_path, result_path, info}

    def _spawn(info):
        fd1, info_path = tempfile.mkstemp(suffix=".json", prefix="proc_info_")
        os.close(fd1)
        fd2, result_path = tempfile.mkstemp(suffix=".json", prefix="proc_result_")
        os.close(fd2)
        with open(info_path, "w") as f:
            json.dump(info, f)
        proc = subprocess.Popen(
            [sys.executable, worker_script, info_path, result_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        return {
            "proc":        proc,
            "info_path":   info_path,
            "result_path": result_path,
            "info":        info,
        }

    def _reap(item):
        proc = item["proc"]
        info = item["info"]
        result_path = item["result_path"]
        info_path = item["info_path"]

        result = None
        if os.path.isfile(result_path):
            try:
                with open(result_path) as f:
                    result = json.load(f)
            except Exception:
                result = None
        if result is None:
            result = {
                'success':   False,
                'file_name': info['file_name'],
                'file':      info['file'],
                'error':     f"worker exited with code {proc.returncode} without producing a result",
            }

        for p in (info_path, result_path):
            try:
                os.remove(p)
            except Exception:
                pass
        return result

    try:
        while pending or active:
            # Fill up the pool, but only as long as memory budget allows.
            # If existing workers are already using >= 4 GB we hold off on
            # spawning more until one finishes and frees memory. This keeps
            # the whole app under the OOM threshold even on large DBT jobs.
            while pending and len(active) < max_workers:
                if active and _workers_memory_bytes(active) >= _MAX_BG_MEMORY_BYTES:
                    break
                active.append(_spawn(pending.pop(0)))

            # Poll workers. 0.3s is a fair balance — fast enough that the
            # progress banner feels live, slow enough that we don't burn CPU
            # spinning while files take minutes to finish.
            time.sleep(0.3)

            still_active = []
            for item in active:
                if item["proc"].poll() is None:
                    still_active.append(item)
                    continue
                result = _reap(item)
                with _BG_LOCK:
                    _BG_JOB["done"] += 1
                    if result.get("success"):
                        _BG_JOB["completed"].append(result["file_name"])
                    else:
                        _BG_JOB["errors"].append(result)
            active = still_active
    finally:
        with _BG_LOCK:
            _BG_JOB["running"] = False
            _BG_JOB["end_time"] = time.time()


# =========================
# Multi-threading Manager Class
# =========================
class BatchProcessingManager:
    """
    Manages multi-threaded batch file processing.
    """
    
    def __init__(self, max_workers=None):
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
        """
        try:
            file = file_info['file']
            file_name = file_info['file_name']
            input_folder = file_info['input_folder']
            final_thresholds = file_info['final_thresholds']
            
            # File paths setup
            original_image_path = os.path.join(input_folder, file)
            output_path = os.path.join(os.getcwd(), 'output', file_name)
            save_dir = os.path.join(output_path, 'dense_mask')

            # Get threshold value
            T = final_thresholds[file_name]

            # Validate file existence
            if not os.path.exists(original_image_path):
                raise FileNotFoundError(f"Original image not found: {original_image_path}")
            if not has_polygon_mask(output_path):
                raise FileNotFoundError("Mask not found for this case.")

            # Load the original volume once — avoids re-reading the file for
            # every slice. load_image returns (volume, affine, shape, img_type).
            image_vol, original_affine, raw_shape, img_type = \
                UnifiedImageLoader.load_image(original_image_path)

            # Prefer the fast 2D polygon format (draw step saves that now).
            # Falls back to loading a legacy 3D dense.nii / dense.dcm if that's
            # all that exists. The 2D mask is reused across every slice — no
            # need to materialise a full 3D mask volume.
            mask_vol = load_polygon_mask_2d(output_path)
            if mask_vol is None:
                is_dicom = os.path.isdir(original_image_path) or original_image_path.lower().endswith((".dcm", ".dicom"))
                mask_path = resolve_dense_mask_path(output_path, prefer_dicom=is_dicom)
                if mask_path is None:
                    raise FileNotFoundError("Mask not found for this case.")
                mask_vol, _, _, _ = UnifiedImageLoader.load_image(mask_path)

            # Re-apply the orientation the user defined in the draw step so
            # the original volume matches the already-oriented mask volume on
            # disk. If the user didn't rotate/flip, this is a no-op.
            orient = _read_orientation_json(output_path)
            slice_dim_raw = int(np.argmin(raw_shape))
            image_vol = apply_orientation(
                image_vol,
                orient["rotate_k"], orient["flip_h"], orient["flip_v"],
                slice_dim=slice_dim_raw,
            )
            original_shape = orient_shape_3d(
                raw_shape, slice_dim_raw, orient["rotate_k"]
            )

            # Save original volume as .nii.gz — rotated 90° CW in the sagittal
            # plane so it displays correctly in ITK-SNAP / 3D Slicer.
            # The mask is rotated the same way in create_final_mask, so they align.
            # image_vol is already float32 from the loaders; rot90 returns a
            # view, so no extra full-volume copy lives in memory here.
            orig_nii_path = os.path.join(output_path, 'original.nii.gz')
            orig_rotated, affine_rotated = sagittal_cw_rotate(
                image_vol, original_affine
            )
            nib.save(
                nib.Nifti1Image(orig_rotated, affine_rotated),
                orig_nii_path,
            )
            print(f"  Original saved as NIfTI (sagittal CW): {orig_nii_path}")

            slice_dim = int(np.argmin(original_shape))
            num_slices = original_shape[slice_dim]
            central_slice_idx = num_slices // 2

            # Extract central slice from the in-memory volume. When mask_vol
            # is the 2D polygon (new fast path), the same slice is used for
            # every slice index — no indexing needed.
            if mask_vol.ndim == 2:
                central_mask_slice = mask_vol.astype(np.float32)
            if slice_dim == 0:
                central_image_slice = image_vol[central_slice_idx].astype(np.float32)
                if mask_vol.ndim != 2:
                    central_mask_slice  = mask_vol[central_slice_idx].astype(np.float32)
            elif slice_dim == 1:
                central_image_slice = image_vol[:, central_slice_idx, :].astype(np.float32)
                if mask_vol.ndim != 2:
                    central_mask_slice  = mask_vol[:, central_slice_idx, :].astype(np.float32)
            else:
                central_image_slice = image_vol[:, :, central_slice_idx].astype(np.float32)
                if mask_vol.ndim != 2:
                    central_mask_slice  = mask_vol[:, :, central_slice_idx].astype(np.float32)

            # Guard against a subtle failure mode: the saved polygon mask
            # was drawn on a slice whose shape no longer matches the image
            # currently being loaded. This happened for some tomosynthesis
            # folders that bundle the DBT multi-frame alongside ancillary
            # 2D DICOMs (projection/synth images) at a different resolution
            # — the draw step used to pick the wrong file. We now pick the
            # multi-frame, but previously-saved masks still have the old
            # shape. Without this check, numpy raises an opaque broadcast
            # error instead of telling the user what to do.
            if central_mask_slice.shape != central_image_slice.shape:
                raise ValueError(
                    f"Saved mask shape {central_mask_slice.shape} does not "
                    f"match image slice shape {central_image_slice.shape}. "
                    f"The polygon was drawn on a different image geometry. "
                    f"Go back to Step 2 and redraw the mask for {file_name}."
                )

            # Calculate target area from central slice
            thresholded_img = ThresholdOperator.threshold_slice(
                central_image_slice, central_mask_slice, T
            )
            target_area = MaskManager.measure_mask_area(thresholded_img)

            print(f"Processing {file_name}:")
            print(f"  Original shape: {original_shape}")
            print(f"  Central slice index: {central_slice_idx}")
            print(f"  Target area: {target_area} pixels")
            print(f"  Slice dimension: {slice_dim}")
            print(f"  Number of slices: {num_slices}")

            # Clean output directory
            if os.path.exists(save_dir):
                for f in os.listdir(save_dir):
                    os.remove(os.path.join(save_dir, f))
            os.makedirs(save_dir, exist_ok=True)

            # Process all slices using the pre-loaded volumes (no more per-slice I/O)
            slice_results = self._process_slices_optimized(
                image_vol, mask_vol, slice_dim, save_dir,
                target_area, num_slices, img_type
            )

            # Free large volumes before final mask assembly
            del image_vol, mask_vol
            gc.collect()
            
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
                print(error_msg)
            return {
                'success': False,
                'file_name': file_name,
                'file': file,
                'error': error_msg
            }
    
    def _process_slices_optimized(self, image_vol, mask_vol, slice_dim, save_dir,
                                 target_area, num_slices, img_type):
        """
        Process all slices from pre-loaded in-memory volumes.
        Volumes are loaded once by the caller to avoid repeated disk I/O.
        Slices are processed in NATIVE orientation and saved as-is.
        """
        total_processed = 0

        compression_level = self.io_settings['compression_level']
        gc_frequency = self.performance_settings['gc_frequency']

        display_transform = DisplayTransform(padding=0)
        display_transform.set_rotation_for_type(img_type)

        # Cumulative per-file timers so we can see where slice-loop time goes.
        t_threshold = 0.0
        t_npy = 0.0
        t_png = 0.0
        t_slice_extract = 0.0
        t_loop_start = time.perf_counter()

        # Fast path: when mask_vol is the 2D polygon, convert once and reuse
        # — the polygon is identical on every slice, so per-slice indexing and
        # per-slice float32 conversion would just be wasted work.
        mask_2d_f32 = mask_vol.astype(np.float32) if mask_vol.ndim == 2 else None

        for slice_index in range(num_slices):
            try:
                # Extract slice from pre-loaded volume — no disk I/O
                t0 = time.perf_counter()
                if slice_dim == 0:
                    image_slice = image_vol[slice_index].astype(np.float32)
                elif slice_dim == 1:
                    image_slice = image_vol[:, slice_index, :].astype(np.float32)
                else:
                    image_slice = image_vol[:, :, slice_index].astype(np.float32)

                if mask_2d_f32 is not None:
                    mask_slice = mask_2d_f32
                elif slice_dim == 0:
                    mask_slice = mask_vol[slice_index].astype(np.float32)
                elif slice_dim == 1:
                    mask_slice = mask_vol[:, slice_index, :].astype(np.float32)
                else:
                    mask_slice = mask_vol[:, :, slice_index].astype(np.float32)
                t_slice_extract += time.perf_counter() - t0

                # Apply dynamic threshold adjustment
                t0 = time.perf_counter()
                adjusted_threshold, thresholded_image = \
                    ThresholdOperator.adjust_slice_threshold(
                        image_slice, mask_slice, target_area
                    )
                t_threshold += time.perf_counter() - t0

                # Convert to binary (0 or 255 for visualization)
                binary_image = np.where(thresholded_image > 0, 255, 0).astype(np.uint8)

                # Save as numpy array (in NATIVE orientation)
                npy_filename = f'slice_{slice_index:04d}_threshold_{adjusted_threshold:.2f}.npy'
                t0 = time.perf_counter()
                np.save(os.path.join(save_dir, npy_filename), binary_image)
                t_npy += time.perf_counter() - t0

                # Save PNG for visualization (rotated for display)
                png_filename = f'slice_{slice_index:04d}_threshold_{adjusted_threshold:.2f}.png'
                if display_transform.rotate_k % 4 != 0:
                    binary_img_display = np.rot90(binary_image, k=display_transform.rotate_k)
                else:
                    binary_img_display = binary_image
                img_pil = Image.fromarray(binary_img_display, mode='L')
                t0 = time.perf_counter()
                img_pil.save(os.path.join(save_dir, png_filename),
                             optimize=True, compress_level=compression_level)
                t_png += time.perf_counter() - t0

                total_processed += 1

                with self.memory_lock:
                    self.operations_count += 1
                    if self.operations_count % gc_frequency == 0:
                        gc.collect()

                del image_slice, mask_slice, thresholded_image, binary_image

            except Exception as e:
                print(f"Error processing slice {slice_index}: {str(e)}")
                continue

        t_wall = time.perf_counter() - t_loop_start
        t_accounted = t_slice_extract + t_threshold + t_npy + t_png
        print(
            f"[timing] slice loop: wall={t_wall:.2f}s over {num_slices} slices | "
            f"threshold={t_threshold:.2f}s ({100 * t_threshold / t_wall:.0f}%) "
            f"extract={t_slice_extract:.2f}s ({100 * t_slice_extract / t_wall:.0f}%) "
            f"npy={t_npy:.2f}s ({100 * t_npy / t_wall:.0f}%) "
            f"png={t_png:.2f}s ({100 * t_png / t_wall:.0f}%) "
            f"other={t_wall - t_accounted:.2f}s"
        )

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
# Global floating status badge
# =========================
@st.fragment(run_every=2)
def render_global_bg_badge():
    """
    Fixed-position status widget pinned to the upper-right corner of the
    viewport. Visible on every page (call from app.py) so the user can
    keep working in Draw / Threshold and still see live processing
    progress. Auto-refreshes every 2s.

    Also syncs finished files into session state so other steps see
    completions in real time without having to re-enter this step.
    """
    with _BG_LOCK:
        running   = _BG_JOB["running"]
        done      = _BG_JOB["done"]
        total     = _BG_JOB["total"]
        completed = list(_BG_JOB["completed"])
        errors    = list(_BG_JOB["errors"])
        start     = _BG_JOB["start_time"]
        end       = _BG_JOB["end_time"]

    if total == 0:
        return  # no job has ever run this session

    # Sync completions into session state for other pages.
    store = st.session_state.get("batch_completed_files", {}).get("process")
    if store is not None:
        for fname in completed:
            if fname not in store:
                store.append(fname)

    pct = int(100 * done / total) if total else 0
    left = total - done
    n_err = len(errors)

    if running:
        elapsed = (time.time() - start) if start else 0
        bg = "#0d6efd"  # blue
        header = f"⚙️ Processing — {pct}%"
        detail = f"{done}/{total} done · {left} left · {elapsed:.0f}s"
    else:
        # Auto-hide the "done" badge 30s after the job finishes so it
        # doesn't stick around forever after navigating away.
        if end is None or (time.time() - end) > 30:
            return
        total_time = (end - start) if (start and end) else 0
        bg = "#198754" if n_err == 0 else "#dc3545"  # green / red
        header = (
            f"✅ Done — {done}/{total}"
            if n_err == 0 else
            f"⚠️ Done · {n_err} error{'s' if n_err != 1 else ''}"
        )
        detail = f"{total_time:.0f}s total"

    # top: 3.75rem keeps the badge clear of Streamlit's Deploy button and
    # hamburger menu (which sit in the top ~3rem strip) while still staying
    # visually anchored to the upper-right corner.
    html = f"""
    <div style="position: fixed; top: 3.75rem; right: 1rem;
                background: {bg}; color: white;
                padding: 0.55rem 0.9rem; border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.25);
                z-index: 9999; font-size: 0.85rem;
                min-width: 220px; max-width: 320px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
      <div style="font-weight: 600; margin-bottom: 0.15rem;">{header}</div>
      <div style="opacity: 0.92;">{detail}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# =========================
# Background progress fragment
# =========================
@st.fragment(run_every=2)
def _render_bg_progress():
    """
    Progress banner for the background job. Fragment-scoped reruns keep the
    rest of the page idle — the user can navigate, draw, and adjust
    thresholds while processing runs.
    """
    with _BG_LOCK:
        running   = _BG_JOB["running"]
        done      = _BG_JOB["done"]
        total     = _BG_JOB["total"]
        completed = list(_BG_JOB["completed"])
        errors    = list(_BG_JOB["errors"])
        start     = _BG_JOB["start_time"]

    # Push newly-finished files into session state so the outer page picks
    # them up on its next natural rerun.
    store = st.session_state.get("batch_completed_files", {}).get("process")
    if store is not None:
        for fname in completed:
            if fname not in store:
                store.append(fname)

    if running:
        st.info(f"⚙️ Background processing running — {done}/{total} files done")
        st.progress(done / total if total else 0)
    else:
        elapsed = (time.time() - start) if start else 0
        st.success(
            f"✅ Background processing complete — {len(completed)}/{total} "
            f"files in {elapsed:.1f}s"
        )
        if errors:
            with st.expander(f"⚠️ {len(errors)} file(s) failed"):
                for e in errors:
                    st.code(e.get("error", str(e)))


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
    # Sync background job → session state
    # =========================
    with _BG_LOCK:
        bg_running   = _BG_JOB["running"]
        bg_done      = _BG_JOB["done"]
        bg_total     = _BG_JOB["total"]
        bg_completed = list(_BG_JOB["completed"])
        bg_errors    = list(_BG_JOB["errors"])
        bg_start     = _BG_JOB["start_time"]

    for fname in bg_completed:
        if fname not in st.session_state["batch_completed_files"]["process"]:
            st.session_state["batch_completed_files"]["process"].append(fname)

    # Recompute after sync
    completed_process          = st.session_state["batch_completed_files"]["process"]
    completed_process_filtered = [f for f in completed_process if f in batch_files_without_extension]
    files_to_process = [
        f for f in batch_files
        if f.split('.')[0] in completed_threshold
        and f.split('.')[0] not in completed_process
    ]

    # Background job status banner — uses st.fragment so only this section
    # reruns every 2s, leaving the rest of the page and navigation responsive
    # (the previous time.sleep(2)+st.rerun() froze the whole script).
    if bg_running or bg_total > 0:
        _render_bg_progress()
        st.divider()

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
    default_workers = min(
        performance_config.get_optimal_workers(len(files_to_process)),
        _MAX_BG_WORKERS,
    )

    max_workers = st.number_input(
        "Number of parallel workers:",
        min_value=1,
        max_value=_MAX_BG_WORKERS,
        value=default_workers,
        step=1,
        help=(
            f"Capped at {_MAX_BG_WORKERS} workers and "
            f"{_MAX_BG_MEMORY_BYTES // (1024**3)} GB total RAM to avoid OOM "
            "on large DBT volumes. The dispatcher also holds off on spawning "
            "new workers when existing ones are already near the RAM cap."
        ),
    )
    
    # Background (non-blocking) button — user can navigate away while it runs
    bg_btn_disabled = bg_running
    if st.button(
        "🚀 Start Background Processing  (navigate freely while running)",
        type="primary",
        disabled=bg_btn_disabled,
        use_container_width=True,
        help="Launches processing in a background thread. You can switch to the Draw or Threshold step while it runs.",
    ):
        with _BG_LOCK:
            _BG_JOB.update({
                "running":    True,
                "completed":  [],
                "errors":     [],
                "total":      len(files_to_process),
                "done":       0,
                "start_time": time.time(),
                "end_time":   None,
            })
        t = threading.Thread(
            target=_run_bg_processing,
            args=(files_to_process, input_folder, final_thresholds, max_workers),
            daemon=True,
        )
        t.start()
        st.toast("Background processing started! Navigate freely — come back here to check progress.", icon="🚀")
        st.rerun()

    st.divider()

    if st.button("⏳ Process All Files (stay on page until done)", use_container_width=True):
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
