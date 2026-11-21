import os
import json
import numpy as np
import streamlit as st
from PIL import Image
from ImageLoader import UnifiedImageLoader
from new_utils import (
    ImageProcessor,
    DisplayTransform,
    apply_orientation,
    load_polygon_mask_2d,
    has_polygon_mask,
)
# Reuse the subprocess-backed prefetch cache from the draw step. Same slow
# call (pydicom decompresses the whole DBT volume for one slice) — so the
# same prefetch machinery applies unchanged. Module-level state is shared
# across steps because Python imports are cached, which is what we want.
from batch_draw_step import _prefetch, _load_slice_prefetched

# How many upcoming pending files to warm in the background after loading
# the current one. Each subprocess decode peaks ~1-2 GB RAM transiently,
# so 2 is a conservative "few" that won't stack up on modest machines.
_THRESHOLD_PREFETCH_AHEAD = 2


def _load_orientation(output_path: str, file_name: str) -> dict:
    """
    Orientation lookup order:
      1. Session state (set during the current run by the draw step)
      2. orientation.json written by the draw step on save
      3. Identity (rotate_k=0, no flips)
    """
    session = st.session_state.get("batch_orientations", {})
    orient = session.get(file_name)
    if orient is not None:
        return {
            "rotate_k": int(orient.get("rotate_k", 0)) % 4,
            "flip_h": bool(orient.get("flip_h", False)),
            "flip_v": bool(orient.get("flip_v", False)),
        }
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


def _ensure_2d(arr):
    """
    Guarantee a 2D slice. Some loader paths (4D NIfTI, odd DICOM layouts)
    return a 3D slice; squeeze size-1 dims and, if still 3D, take the
    middle index along the smallest axis — same rule used by the draw step.
    """
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim == 3:
        slice_dim = int(np.argmin(arr.shape))
        idx = arr.shape[slice_dim] // 2
        if slice_dim == 0:
            arr = arr[idx]
        elif slice_dim == 1:
            arr = arr[:, idx, :]
        else:
            arr = arr[:, :, idx]
    return arr


@st.cache_data(show_spinner=False)
def _load_display_data(original_image_path, output_path, rotate_k: int, flip_h: bool, flip_v: bool, mask_version: int = 0):
    """Load + normalise images once per (file, orientation); cached so the slider doesn't re-read disk."""
    # _load_slice_prefetched returns from the shared subprocess cache when the
    # file was warmed by a previous rerun; otherwise it loads synchronously.
    img, _, _, _, img_type = _load_slice_prefetched(original_image_path)
    msk = load_polygon_mask_2d(output_path)
    if msk is None:
        raise FileNotFoundError(f"No saved polygon mask found for {output_path}")
    img = _ensure_2d(img)
    msk = _ensure_2d(msk)
    # The mask on disk is already in the oriented frame (draw step saved it
    # that way); only the original image needs the user orientation applied.
    img = apply_orientation(img, rotate_k, flip_h, flip_v)
    img_show = ImageProcessor.normalize_image(img)          # uint8 0-255
    msk_show = (msk > 0).astype(np.uint8)                   # binary
    dt = DisplayTransform(padding=0)
    dt.set_rotation_for_type(img_type)
    if dt.rotate_k % 4 != 0:
        img_show = np.rot90(img_show, k=dt.rotate_k)
        msk_show = np.rot90(msk_show, k=dt.rotate_k)
    return img_show, msk_show, img_type


def _render_threshold_pil(img_show, msk_show, threshold, width):
    """
    Fast PIL-based threshold overlay.
    Replaces matplotlib figure rendering (was ~4800×4800 px at dpi=300).
    Thresholded pixels get a red tint at 35% alpha; everything else stays grey.
    """
    norm         = img_show.astype(np.float32) / 255.0
    thresh_region = (norm > threshold) & (msk_show > 0)

    rgb = np.stack([img_show, img_show, img_show], axis=-1).astype(np.float32)
    rgb[thresh_region, 0] = rgb[thresh_region, 0] * 0.65 + 255 * 0.35  # red
    rgb[thresh_region, 1] = rgb[thresh_region, 1] * 0.65
    rgb[thresh_region, 2] = rgb[thresh_region, 2] * 0.65
    result = rgb.clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(result)
    h, w    = result.shape[:2]
    if w > 0:
        pil_img = pil_img.resize((width, max(1, int(width * h / w))), Image.LANCZOS)
    return pil_img


def batch_threshold_step():
    """
    Step 3: Batch threshold adjustment for all files.
    Organizes navigation, threshold selection, and saving for each file in the batch.
    """

    # =====================================================
    # UI header and styling
    # =====================================================
    with open("static/batch_threshold_step.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # =====================================================
    # Session state and Data validation
    # =====================================================
    if "batch_files" not in st.session_state:
        st.error("No batch files selected. Please go back to file selection.")
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        return

    batch_files = st.session_state["batch_files"]
    current_index = st.session_state.get("batch_current_index", 0)
    completed_draw = st.session_state["batch_completed_files"]["draw"]
    all_completed_threshold = st.session_state["batch_completed_files"]["threshold"]
    completed_threshold = [f for f in all_completed_threshold if f in st.session_state["batch_files_without_extension"]] # only the set of batch files that already has a polygon
    
    # check if the threshold.json is in the output folder
    for file in completed_threshold:
        output_path = os.path.join(os.getcwd(), 'output', file)
        threshold_json = os.path.join(output_path, "threshold.json")
        if not os.path.exists(threshold_json):
            completed_threshold.remove(file)

    # =====================================================
    # Draw step completion check
    # =====================================================
    if len(completed_draw) == 0:
        st.error("No masks drawn yet. Please draw at least one mask first.")
        if st.button("← Back to Draw Step"):
            st.session_state["current_step"] = "batch_draw"
            st.rerun()
        return



    # =====================================================
    # Draw-then-threshold flow: if we were sent here from the draw step
    # after saving a single mask, bounce straight back to draw for the
    # next file WITHOUT showing any completion screen.
    # =====================================================
    draw_return = st.session_state.get("draw_return_index")
    if draw_return is not None:
        # Only redirect AFTER the threshold for the current file is saved.
        # If it isn't saved yet, fall through and show the threshold UI.
        current_file_for_check = batch_files[current_index] if current_index < len(batch_files) else None
        current_file_name_for_check = current_file_for_check.split('.')[0] if current_file_for_check else None
        if current_file_name_for_check and current_file_name_for_check in completed_threshold:
            st.session_state.pop("draw_return_index", None)
            st.session_state["batch_current_index"] = draw_return
            st.session_state["current_step"] = "batch_draw"
            st.rerun()

    # =====================================================
    # Progress bar and overall navigation
    # =====================================================
    # Use only files that have a mask as the denominator
    files_with_mask = len(completed_draw)
    st.markdown('</div>', unsafe_allow_html=True)

    # "Done" when every masked file also has a threshold — show completion message
    batch_files_without_ext = st.session_state.get("batch_files_without_extension", [])
    all_done = (
        all(f in completed_draw for f in batch_files_without_ext)
        and all(f in completed_threshold for f in batch_files_without_ext)
    )
    if all_done:
        st.success("🎉 All files masked and thresholded! Ready to process.")
        st.session_state["current_step"] = "batch_process"
        st.rerun()

    # =====================================================
    # File selection and skipping logic 
    # =====================================================
    current_file = batch_files[current_index]
    current_file_name = current_file.split('.')[0]

    # Skip if already completed: go to next pending file
    if current_file_name in completed_threshold:
        next_index = None
        for idx in range(current_index + 1, len(batch_files)):
            if batch_files[idx].split('.')[0] not in completed_threshold:
                next_index = idx
                break
        if next_index is not None:
            st.session_state["batch_current_index"] = next_index
        else:
            st.session_state["batch_current_index"] = len(batch_files)
        st.rerun()
        return

    # Skip if not in completed_draw: go to next file
    if current_file_name not in completed_draw:
        st.session_state["batch_current_index"] = current_index + 1
        st.rerun()
        return

    # =====================================================
    # Current file UI and navigation 
    # =====================================================

    # =====================================================
    # Completed files list 
    # =====================================================
    # =====================================================
    # File paths for current file 
    # =====================================================
    # Use main input folder selected in Step 1
    input_folder = st.session_state.get("main_input_folder", os.path.join(os.getcwd(), "media"))
    output_path = os.path.join(os.getcwd(), "output", current_file_name)
    original_image_path = os.path.join(input_folder, current_file)

    if not os.path.exists(input_folder):
        st.error(f"The selected main input folder does not exist: {input_folder}")
        return


    # =====================================================
    # File existence and image loading
    # =====================================================
    if not has_polygon_mask(output_path):
        st.error("Mask file not found for this case.")
        st.write("This file may not have completed the draw step properly.")
        if st.button("Skip this file"):
            st.session_state["batch_current_index"] = current_index + 1
            st.rerun()
        return

    try:
        orient = _load_orientation(output_path, current_file_name)
        mask_version = st.session_state.get("mask_version", {}).get(current_file_name, 0)
        img_show, msk_show, img_type = _load_display_data(
            original_image_path, output_path,
            orient["rotate_k"], orient["flip_h"], orient["flip_v"],
            mask_version,
        )
    except Exception as e:
        st.error(f"Error loading images: {str(e)}")
        if st.button("Skip this file"):
            st.session_state["batch_current_index"] = current_index + 1
            st.rerun()
        return

    # Warm the next few pending files in background subprocesses so that
    # clicking Save & Continue / Next → lands on an already-decoded slice.
    # Guarded per index: every slider / number-input interaction reruns the
    # script, but we only want to kick the prefetch once per displayed file.
    prefetch_key = f"_threshold_prefetch_idx_{current_index}"
    if not st.session_state.get(prefetch_key):
        warmed = 0
        for idx in range(current_index + 1, len(batch_files)):
            if warmed >= _THRESHOLD_PREFETCH_AHEAD:
                break
            nxt_name = batch_files[idx].split('.')[0]
            if nxt_name in completed_draw and nxt_name not in completed_threshold:
                _prefetch(os.path.join(input_folder, batch_files[idx]))
                warmed += 1
        st.session_state[prefetch_key] = True

    # =====================================================
    # Sidebar: Threshold control (stays visible while reviewing images)
    # =====================================================
    if "batch_thresholds" not in st.session_state:
        st.session_state["batch_thresholds"] = {}

    def update_threshold():
        st.session_state["batch_thresholds"][current_file_name] = st.session_state[f"input_{current_file_name}"]

    with st.sidebar:
        st.markdown("### 🎯 Threshold")
        threshold = st.number_input(
            "Threshold value",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state["batch_thresholds"].get(current_file_name, 0.38)),
            step=0.01,
            format="%.2f",
            key=f"input_{current_file_name}",
            help="Lower = more tissue included, Higher = more selective",
            on_change=update_threshold,
        )
        st.session_state["batch_thresholds"][current_file_name] = threshold
        st.caption("Lower value = more tissue included")

        if st.button("💾 Save & Continue", type="primary", use_container_width=True):
            if "batch_final_thresholds" not in st.session_state:
                st.session_state["batch_final_thresholds"] = {}
            st.session_state["batch_final_thresholds"][current_file_name] = threshold
            try:
                threshold_json = os.path.join(output_path, "threshold.json")
                with open(threshold_json, "w") as f:
                    json.dump({"threshold": threshold}, f)
                st.session_state["batch_completed_files"]["threshold"].append(current_file_name)
                st.success(f"Threshold {threshold:.2f} saved successfully!")
            except Exception as e:
                st.error(f"Could not save threshold: {e}")
            # Rerun: the top-of-page draw_return_index check will bounce
            # back to the draw step if we're in one-file-at-a-time mode.
            st.rerun()

        if st.button("← Back to Draw Step", use_container_width=True):
            # Remove from completed so the draw step opens directly in draw
            # mode (not view mode), and the threshold can be re-set after.
            st.session_state["batch_completed_files"]["draw"] = [
                f for f in st.session_state["batch_completed_files"]["draw"]
                if f != current_file_name
            ]
            st.session_state["batch_completed_files"]["threshold"] = [
                f for f in st.session_state["batch_completed_files"]["threshold"]
                if f != current_file_name
            ]
            st.session_state["current_step"] = "batch_draw"
            st.rerun()

        # Allow the user to stop early and process only the files
        # that have already been annotated and thresholded.
        n_ready = len(completed_threshold)
        n_total = len(batch_files)
        if n_ready > 0 and n_ready < n_total:
            st.markdown("---")
            if st.button(
                f"⚡ Process {n_ready}/{n_total} Completed Files",
                use_container_width=True,
                help=f"Stop annotating and run processing on the {n_ready} file(s) already finished.",
            ):
                st.session_state["current_step"] = "batch_process"
                st.rerun()

    # =====================================================
    # Image preview
    # =====================================================
    comparison_mode = True
    selected_width  = 800

    # Image preview section — pre-compute both images before rendering any
    # columns so the browser receives them together and the user never sees
    # the left column resize while the right one is still being computed.

    col_w = int(selected_width * 0.55)
    h, w  = img_show.shape
    orig_pil   = Image.fromarray(img_show).resize(
        (col_w, max(1, int(col_w * h / w))), Image.LANCZOS
    )
    thresh_pil = _render_threshold_pil(img_show, msk_show, threshold, col_w)

    if comparison_mode:
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown("**Original Image**")
            st.image(orig_pil, width=col_w)
        with img_col2:
            st.markdown("**Thresholded Result**")
            st.image(thresh_pil, width=col_w)
    else:
        _, img_col, _ = st.columns([1, 2, 1])
        thresh_pil_single = _render_threshold_pil(img_show, msk_show, threshold, selected_width)
        with img_col:
            st.image(thresh_pil_single, width=selected_width)


