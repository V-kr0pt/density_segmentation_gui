# =========================
# Imports
# =========================
import os
import sys
import json
import pickle
import tempfile
import threading
import subprocess
from collections import OrderedDict

import streamlit as st
import numpy as np
import plotly.express as px
from streamlit_plotly_events import plotly_events

from new_utils import (
    MaskManager,
    DisplayTransform,
    apply_orientation,
    orient_shape_3d,
    save_polygon_mask_dcm,
    load_polygon_mask_2d,
)
from ImageLoader import UnifiedImageLoader


# =====================================================================
# Background prefetch
# =====================================================================
# Loading the next DBT file costs 10-20s because pydicom decompresses the
# whole multi-frame volume even when we only want the central slice, and
# the decode path can hold the GIL long enough to stall Streamlit's UI.
#
# To keep the polygon canvas responsive while the next file loads we run
# the prefetch in a SEPARATE Python process (via subprocess). The child
# has its own GIL, so its decode work can't block the streamlit thread
# no matter what pydicom does internally. Results are serialized to a
# temp pickle and picked up by the parent on the next rerun.
#
# Module-level state persists across Streamlit reruns (the script re-runs
# but imports are cached), so the cache + in-flight dict survive between
# user interactions. Access is serialized by _PREFETCH_LOCK.
_PREFETCH_LOCK = threading.Lock()
_PREFETCH_CACHE: "OrderedDict[str, tuple]" = OrderedDict()
# file_path -> (subprocess.Popen, result_tmp_path)
_PREFETCH_PENDING: dict = {}
# Cache is shared with batch_threshold_step (which prefetches up to 2 ahead),
# so size needs to hold: current + 2 ahead + previous + draw-step entry.
_PREFETCH_MAX = 6
# Cap concurrent prefetch subprocesses. Each DBT decode can use ~1-2 GB RAM;
# previously draw (1-ahead) + threshold (2-ahead) could spawn 3 at once and
# thrash, causing the threshold step to hang on proc.wait.
_PREFETCH_CONCURRENCY = 2


def _harvest_pending() -> None:
    """Move any completed subprocess results into the cache and clean up temp files."""
    with _PREFETCH_LOCK:
        pending_items = list(_PREFETCH_PENDING.items())

    for path, (proc, tmp) in pending_items:
        if proc.poll() is None:
            continue  # still running
        # Try to read the pickled result
        result = None
        if os.path.isfile(tmp):
            try:
                with open(tmp, "rb") as f:
                    result = pickle.load(f)
            except Exception:
                result = None
            try:
                os.remove(tmp)
            except Exception:
                pass
        err_path = tmp + ".err"
        if os.path.isfile(err_path):
            try:
                os.remove(err_path)
            except Exception:
                pass
        with _PREFETCH_LOCK:
            _PREFETCH_PENDING.pop(path, None)
            if result is not None:
                _PREFETCH_CACHE[path] = result
                while len(_PREFETCH_CACHE) > _PREFETCH_MAX:
                    _PREFETCH_CACHE.popitem(last=False)


def _prefetch(file_path: str) -> None:
    """Launch a subprocess to load file_path, unless already cached or in flight."""
    if not file_path:
        return
    _harvest_pending()
    with _PREFETCH_LOCK:
        if file_path in _PREFETCH_CACHE or file_path in _PREFETCH_PENDING:
            return
        if len(_PREFETCH_PENDING) >= _PREFETCH_CONCURRENCY:
            return  # at concurrency cap; skip warming this one
        fd, tmp = tempfile.mkstemp(suffix=".pkl", prefix="prefetch_")
        os.close(fd)
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_prefetch_worker.py")
        # CREATE_NO_WINDOW avoids a console flash on Windows; harmless elsewhere.
        creationflags = 0x08000000 if os.name == "nt" else 0
        proc = subprocess.Popen(
            [sys.executable, worker, file_path, tmp],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        _PREFETCH_PENDING[file_path] = (proc, tmp)


def _load_slice_prefetched(file_path: str):
    """Return a cached slice if prefetched; otherwise load sync and cache it."""
    _harvest_pending()
    with _PREFETCH_LOCK:
        cached = _PREFETCH_CACHE.get(file_path)
    if cached is not None:
        return cached
    # If a prefetch subprocess is already running for this file, wait for
    # it rather than launching a duplicate sync load (which would double
    # the disk I/O).
    with _PREFETCH_LOCK:
        pending = _PREFETCH_PENDING.get(file_path)
    if pending is not None:
        proc, tmp = pending
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            # Subprocess is stuck; kill it so a later rerun doesn't re-block
            # on the same entry, then fall through to a sync load below.
            try:
                proc.kill()
            except Exception:
                pass
            with _PREFETCH_LOCK:
                _PREFETCH_PENDING.pop(file_path, None)
        except Exception:
            pass
        _harvest_pending()
        with _PREFETCH_LOCK:
            cached = _PREFETCH_CACHE.get(file_path)
        if cached is not None:
            return cached
    # Cold path — load synchronously on the main thread.
    result = UnifiedImageLoader.load_slice(file_path)
    with _PREFETCH_LOCK:
        _PREFETCH_CACHE[file_path] = result
        while len(_PREFETCH_CACHE) > _PREFETCH_MAX:
            _PREFETCH_CACHE.popitem(last=False)
    return result


_DEFAULT_ORIENT = {"rotate_k": 0, "flip_h": False, "flip_v": False}


def _orient_keys(file_name: str) -> tuple[str, str, str]:
    """Per-file widget state keys for rotate/flip. Dashes break some Streamlit widget keys."""
    safe = file_name.replace(".", "_").replace("/", "_").replace("\\", "_")
    return f"orient_rot_k_{safe}", f"orient_fh_{safe}", f"orient_fv_{safe}"


def _current_orientation(file_name: str) -> dict:
    """Read orientation from widget state keys (source of truth)."""
    rot_key, fh_key, fv_key = _orient_keys(file_name)
    return {
        "rotate_k": int(st.session_state.get(rot_key, 0)) % 4,
        "flip_h": bool(st.session_state.get(fh_key, False)),
        "flip_v": bool(st.session_state.get(fv_key, False)),
    }


def _drop_canvas_state():
    """
    Wipe drawing/cache state. Safe to call from a widget callback.
    Polygon-tracking keys are RESET to empty containers (not popped), because
    the click handler assumes they exist once the per-file init block has run.
    """
    for k in ("mask", "result", "create_mask",
              "_disp_cache_key", "_disp_img_rgb", "_disp_w", "_disp_h"):
        st.session_state.pop(k, None)
    st.session_state["current_points"] = []
    st.session_state["completed_polygons_display"] = []
    st.session_state["native_polygons"] = []
    st.session_state["last_click"] = None


def _on_orient_widget_change():
    """on_change for Flip H/Flip V checkboxes — just clear the canvas state."""
    _drop_canvas_state()


def _on_rotate_click(rot_key: str):
    st.session_state[rot_key] = (int(st.session_state.get(rot_key, 0)) + 3) % 4
    _drop_canvas_state()


def _on_reset_click(rot_key: str, fh_key: str, fv_key: str):
    st.session_state[rot_key] = 0
    st.session_state[fh_key] = False
    st.session_state[fv_key] = False
    _drop_canvas_state()


def _load_orientation_from_disk(output_path: str) -> dict | None:
    """Read orientation.json from a completed case, if present."""
    path = os.path.join(output_path, "orientation.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return {
            "rotate_k": int(data.get("rotate_k", 0)) % 4,
            "flip_h": bool(data.get("flip_h", False)),
            "flip_v": bool(data.get("flip_v", False)),
        }
    except Exception:
        return None


def _save_orientation_to_disk(output_path: str, orient: dict) -> None:
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "orientation.json"), "w") as f:
        json.dump(
            {
                "rotate_k": int(orient["rotate_k"]) % 4,
                "flip_h": bool(orient["flip_h"]),
                "flip_v": bool(orient["flip_v"]),
            },
            f,
        )




def _get_display_rgb(cache_key, image_slice, img_type, transform):
    if st.session_state.get("_disp_cache_key") == cache_key:
        return (
            st.session_state["_disp_img_rgb"],
            st.session_state["_disp_w"],
            st.session_state["_disp_h"],
        )
    pil_image = transform.prepare_for_display(
        image_slice, img_type=img_type, max_width=1200, max_height=800
    )
    display_w, display_h = pil_image.size
    arr = np.array(pil_image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    st.session_state["_disp_cache_key"] = cache_key
    st.session_state["_disp_img_rgb"]   = arr
    st.session_state["_disp_w"]         = display_w
    st.session_state["_disp_h"]         = display_h
    return arr, display_w, display_h


def _clear_draw_state():
    keys = [
        "display_transform", "mask", "result", "create_mask",
        "affine", "original_image_path", "slice_index", "output_path",
        "original_shape", "current_batch_file", "img_type",
        "native_polygons", "completed_polygons_display",
        "current_points", "last_click", "canvas_generation",
        "_disp_cache_key", "_disp_img_rgb", "_disp_w", "_disp_h",
    ]
    for k in keys:
        st.session_state.pop(k, None)


_STROKE = ["red", "lime", "royalblue", "gold", "magenta", "cyan", "orange"]
_FILL   = [
    "rgba(255,0,0,0.20)",   "rgba(0,255,0,0.20)",     "rgba(65,105,225,0.20)",
    "rgba(255,215,0,0.20)", "rgba(255,0,255,0.20)",   "rgba(0,255,255,0.20)",
    "rgba(255,165,0,0.20)",
]


def _build_figure(img_rgb, completed_display, current_points, display_w, display_h):
    # px.imshow with a 3-D RGB array produces go.Image - a real Plotly trace
    # that fires click events, unlike go.layout.Image which is a layout element.
    fig = px.imshow(img_rgb, aspect="equal")
    fig.update_layout(
        height=display_h, width=display_w,
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False,
        dragmode=False,
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)
    # Hide the x/y tooltip on the image but keep click events.
    # hoverinfo="none" = no label, events still fire.
    # hoverinfo="skip" would disable the click pipeline entirely — don't use it here.
    # hovermode=False at layout level also blocks clicks, so avoid that too.
    fig.update_traces(hoverinfo="none", hovertemplate=None, selector=dict(type="image"))

    for i, poly in enumerate(completed_display):
        if len(poly) < 2:
            continue
        c  = _STROKE[i % len(_STROKE)]
        fc = _FILL[i % len(_FILL)]
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        fig.add_scatter(x=xs, y=ys, mode="lines",
                        line=dict(color=c, width=2),
                        fill="toself", fillcolor=fc,
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=[p[0] for p in poly], y=[p[1] for p in poly],
                        mode="markers", marker=dict(size=8, color=c),
                        showlegend=False, hoverinfo="skip")
        cx_a = sum(p[0] for p in poly) / len(poly)
        cy_a = sum(p[1] for p in poly) / len(poly)
        fig.add_annotation(x=cx_a, y=cy_a, text=f"#{i+1}",
                           font=dict(color=c, size=14), showarrow=False)

    if current_points:
        fig.add_scatter(
            x=[p[0] for p in current_points],
            y=[p[1] for p in current_points],
            mode="lines+markers",
            line=dict(color="yellow", width=2, dash="dot"),
            marker=dict(size=10, color="yellow", symbol="circle-open",
                        line=dict(width=2, color="yellow")),
            showlegend=False, hoverinfo="skip",
        )
    return fig


def batch_draw_step():
    with open("static/batch_draw_step.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    if "batch_files" not in st.session_state:
        st.error("No batch files selected. Please go back to file selection.")
        if st.button("Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        return

    batch_files        = st.session_state["batch_files"]
    current_index      = st.session_state.get("batch_current_index", 0)
    all_completed_draw = st.session_state["batch_completed_files"]["draw"]
    completed_draw     = [
        f for f in all_completed_draw
        if f in st.session_state["batch_files_without_extension"]
    ]

    total_files = len(batch_files)

    if current_index >= len(batch_files):
        completed_threshold = st.session_state["batch_completed_files"]["threshold"]
        all_thresholded = all(
            f in completed_threshold
            for f in st.session_state["batch_files_without_extension"]
        )
        if all_thresholded:
            st.success("All masks drawn and thresholds set! Ready to process.")
            if st.button("→ Go to Process Step", type="primary", use_container_width=True):
                # st.session_state.pop("draw_return_index", None)
                st.session_state["current_step"] = "batch_process"
                st.rerun()
        else:
            st.success("All masks have been drawn!")
            if st.button("Continue to Threshold Step"):
                st.session_state.pop("draw_return_index", None)
                st.session_state["batch_current_index"] = 0
                st.session_state["current_step"] = "batch_threshold"
                st.rerun()
        if st.button("Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        return

        # Allow jumping to threshold step whenever at least one mask exists
    # if len(completed_draw) > 0:
    #     if st.button(
    #         f"→ Go to Threshold Step ({len(completed_draw)} file{'s' if len(completed_draw) != 1 else ''} ready)",
    #         type="primary",
    #         use_container_width=True,
    #     ):
    #         st.session_state.pop("draw_return_index", None)  # manual jump: free navigation
    #         st.session_state["batch_current_index"] = 0
    #         st.session_state["current_step"] = "batch_threshold"
    #         st.rerun()

    current_file      = batch_files[current_index]
    current_file_name = current_file.split(".")[0]
    is_completed      = current_file_name in completed_draw

    # ---- Shared: load original image ----
    input_folder = st.session_state.get(
        "main_input_folder", os.path.join(os.getcwd(), "media")
    )
    file_path = os.path.join(input_folder, current_file)

    if not os.path.exists(input_folder):
        st.error(f"Input folder not found: {input_folder}")
        return
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return

    with st.spinner("Loading image..."):
        image_slice, affine, original_shape, slice_index, img_type = (
            _load_slice_prefetched(file_path)
        )

    # Warm the NEXT file in the background — exactly once per current
    # file. Every landmark click causes a Streamlit rerun; without this
    # session-state guard we'd re-enter _prefetch on every click and,
    # even though _PREFETCH_PENDING prevents duplicate spawns, the guard
    # makes the "do it once" behavior explicit and cheap. The subprocess
    # itself has its own GIL so pydicom decompression there can't freeze
    # this UI thread.
    prefetch_key = f"_prefetch_kicked_idx_{current_index}"
    if not st.session_state.get(prefetch_key):
        nxt = current_index + 1
        if nxt < len(batch_files):
            next_file_path = os.path.join(input_folder, batch_files[nxt])
            _prefetch(next_file_path)
        st.session_state[prefetch_key] = True

    # =========================================================
    # VIEW MODE  — file already has a saved mask
    # =========================================================
    if is_completed:
        output_path = os.path.join(os.getcwd(), "output", current_file_name)

        # The saved mask is already in the oriented frame; apply the same
        # orientation to the raw original so they line up on screen.
        view_orient = (
            _load_orientation_from_disk(output_path)
            or dict(_DEFAULT_ORIENT)
        )
        view_image = apply_orientation(
            image_slice,
            view_orient["rotate_k"], view_orient["flip_h"], view_orient["flip_v"],
        )

        mask_slice = load_polygon_mask_2d(output_path)

        if mask_slice is None:
            st.warning("Saved mask file not found on disk. You can redraw it below.")
        else:
            # Surface shape mismatch early. Some tomosynthesis folders bundle
            # the DBT multi-frame with an ancillary 2D DICOM at a different
            # resolution; earlier builds of the loader picked the ancillary
            # file so the saved polygon has the wrong geometry for the DBT
            # volume that the process step will use. Catching it here saves
            # the user from a cryptic failure later in Step 4.
            if mask_slice.shape != view_image.shape:
                st.warning(
                    f"Saved mask shape {mask_slice.shape} doesn't match the "
                    f"current image shape {view_image.shape}. The mask was "
                    f"drawn on a different image — click **Redraw Mask** "
                    f"below to fix before running Step 4."
                )
            st.markdown("### Saved Mask Preview")
            col1, col2 = st.columns(2)

            view_transform = DisplayTransform(padding=50)

            present_orig = view_transform.prepare_for_display(
                view_image, img_type=img_type, max_width=900, max_height=600
            )
            present_mask = view_transform.prepare_for_display(
                mask_slice, img_type=img_type, max_width=900, max_height=600
            )

            with col1:
                st.image(present_orig, caption="Original Image",  use_container_width=True)
            with col2:
                st.image(present_mask, caption="Saved Mask",      use_container_width=True)

        st.divider()

        col_redraw, col_next = st.columns(2)
        with col_redraw:
            if st.button("✏️ Redraw Mask", type="primary", use_container_width=True,
                         help="Delete the saved mask and draw a new one"):
                # Remove from completed lists so drawing is unlocked
                st.session_state["batch_completed_files"]["draw"] = [
                    f for f in st.session_state["batch_completed_files"]["draw"]
                    if f != current_file_name
                ]
                # Also clear downstream completed states for this file
                for step in ("threshold", "process"):
                    st.session_state["batch_completed_files"][step] = [
                        f for f in st.session_state["batch_completed_files"][step]
                        if f != current_file_name
                    ]
                _clear_draw_state()
                st.rerun()
        with col_next:
            next_idx = current_index + 1
            if next_idx < len(batch_files) and st.button("Next →", key="view_next",
                                                          use_container_width=True):
                st.session_state["batch_current_index"] = next_idx
                _clear_draw_state()
                st.rerun()
        return

    # =========================================================
    # DRAW MODE  — file not yet completed
    # =========================================================

    # Initialize per-file orientation widget state BEFORE any widget renders.
    # If the user is redrawing a completed file, seed from orientation.json.
    rot_key, fh_key, fv_key = _orient_keys(current_file_name)
    if rot_key not in st.session_state:
        output_path_for_seed = os.path.join(os.getcwd(), "output", current_file_name)
        disk_orient = _load_orientation_from_disk(output_path_for_seed)
        if disk_orient is not None:
            st.session_state[rot_key] = disk_orient["rotate_k"]
            st.session_state[fh_key] = disk_orient["flip_h"]
            st.session_state[fv_key] = disk_orient["flip_v"]
        else:
            st.session_state[rot_key] = 0
            st.session_state[fh_key] = False
            st.session_state[fv_key] = False

    orient = _current_orientation(current_file_name)
    raw_shape = original_shape
    slice_dim_raw = int(np.argmin(raw_shape)) if len(raw_shape) == 3 else 0
    oriented_shape = (
        orient_shape_3d(raw_shape, slice_dim_raw, orient["rotate_k"])
        if len(raw_shape) == 3 else raw_shape
    )
    oriented_image = apply_orientation(
        image_slice, orient["rotate_k"], orient["flip_h"], orient["flip_v"]
    )

    if st.session_state.get("current_batch_file") != current_file:
        st.session_state.update({
            "affine":                     affine,
            "original_image_path":        file_path,
            "original_shape":             oriented_shape,
            "slice_index":                slice_index,
            "current_batch_file":         current_file,
            "img_type":                   img_type,
            "native_polygons":            [],
            "completed_polygons_display": [],
            "current_points":             [],
            "last_click":                 None,
        })
        for k in ("mask", "result", "create_mask", "display_transform",
                  "_disp_cache_key", "_disp_img_rgb", "_disp_w", "_disp_h"):
            st.session_state.pop(k, None)
    else:
        # Keep oriented shape in sync if the user flipped/rotated
        st.session_state["original_shape"] = oriented_shape

    if "display_transform" not in st.session_state:
        st.session_state["display_transform"] = DisplayTransform(padding=50)
    transform = st.session_state["display_transform"]

    # Cache display by (file, orientation) so flip/rotate invalidates it
    disp_cache_id = (
        file_path,
        orient["rotate_k"], orient["flip_h"], orient["flip_v"],
    )
    img_rgb, display_w, display_h = _get_display_rgb(
        disp_cache_id, oriented_image, img_type, transform
    )

    completed_display = st.session_state.get("completed_polygons_display", [])
    current_points    = st.session_state.get("current_points", [])
    n_done = len(completed_display)
    n_pts  = len(current_points)

    fig = _build_figure(img_rgb, completed_display, current_points, display_w, display_h)

    # Status caption
    if n_pts == 0 and n_done == 0:
        st.caption("Click anywhere on the image to start placing vertices.")
    elif n_pts > 0:
        need = max(0, 3 - n_pts)
        suffix = f"  Place {need} more to enable Close Polygon." if need else ""
        st.caption(f"{n_pts} vertex{'es' if n_pts != 1 else ''} placed.{suffix}")
    else:
        st.caption(f"{n_done} polygon{'s' if n_done != 1 else ''} completed.")

    # ---- Drawing Canvas ----
    st.markdown("### Drawing Canvas")
    gen = st.session_state.get("canvas_generation", 0)
    canvas_key = "canvas_" + current_file.replace("/", "_").replace("\\", "_").replace(".", "_") + f"_g{gen}"

    clicked = plotly_events(
        fig,
        click_event=True,
        select_event=False,
        hover_event=False,
        override_height=display_h,
        override_width=display_w,
        key=canvas_key,
    )

    # plotly_events replays the last stored event on every rerun;
    # deduplicate so button-triggered reruns don't add duplicate vertices.
    if clicked:
        ev = clicked[-1]
        new_click = (round(float(ev["x"])), round(float(ev["y"])))
        if new_click != st.session_state.get("last_click"):
            st.session_state["last_click"] = new_click
            st.session_state["current_points"].append(new_click)
            st.rerun()

    # ---- Drawing controls in the sidebar (below Current Step) ----
    has_mask = "result" in st.session_state and "mask" in st.session_state
    mask_area = int(np.count_nonzero(st.session_state["mask"])) if has_mask else 0
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ✏️ Draw Controls")
        if st.button("Close Polygon", disabled=n_pts < 3,
                     use_container_width=True,
                     help="Finalise the current polygon (needs >= 3 points)"):
            disp_pts   = list(st.session_state["current_points"])
            native_pts = transform.canvas_to_native_coords(disp_pts)
            st.session_state["completed_polygons_display"].append(disp_pts)
            st.session_state["native_polygons"].append(native_pts)
            st.session_state["current_points"] = []
            st.rerun()

        if st.button("Undo Last Point", disabled=n_pts == 0,
                     use_container_width=True,
                     help="Remove the most recently placed vertex"):
            st.session_state["current_points"].pop()
            st.rerun()

        if st.button("Delete Last Polygon", disabled=n_done == 0,
                     use_container_width=True,
                     help="Remove the most recently closed polygon"):
            st.session_state["completed_polygons_display"].pop()
            st.session_state["native_polygons"].pop()
            st.rerun()

        if st.button("Clear All", use_container_width=True,
                     help="Delete all polygons and in-progress points"):
            st.session_state["current_points"]             = []
            st.session_state["completed_polygons_display"] = []
            st.session_state["native_polygons"]            = []
            st.session_state["last_click"]                 = None
            st.session_state["canvas_generation"]          = st.session_state.get("canvas_generation", 0) + 1
            st.rerun()

        st.markdown("---")

        if st.button("✅ Create Mask & Continue", disabled=n_done == 0,
                     type="primary", use_container_width=True,
                     help="Generate mask from closed polygons, save, and continue"):
            # Build the mask inline (no separate Create Mask step needed)
            valid = [p for p in st.session_state.get("native_polygons", []) if len(p) >= 3]
            if valid:
                with st.spinner("Saving mask…"):
                    result, combined_mask = MaskManager.create_combined_mask(oriented_image, valid)
                    mask_area_now = int(np.count_nonzero(combined_mask))
                    if mask_area_now > 0:
                        output_path_now = os.path.join(os.getcwd(), "output", current_file_name)
                        if current_file_name not in st.session_state["batch_completed_files"]["draw"]:
                            save_polygon_mask_dcm(
                                mask_2d=combined_mask,
                                output_path=output_path_now,
                                source_image_path=st.session_state.get("original_image_path"),
                            )
                            _save_orientation_to_disk(output_path_now, orient)
                            st.session_state["batch_completed_files"]["draw"].append(current_file_name)
                            # Invalidate threshold step's display cache for this file
                            if "mask_version" not in st.session_state:
                                st.session_state["mask_version"] = {}
                            st.session_state["mask_version"][current_file_name] =                                 st.session_state["mask_version"].get(current_file_name, 0) + 1
                if mask_area_now == 0:
                    st.error("Mask is empty — clear points and re-draw.")
                else:
                    st.session_state["batch_current_index"] = current_index
                    st.session_state["draw_return_index"] = current_index + 1
                    _clear_draw_state()
                    st.toast(f"Mask saved for {current_file}!", icon="✅")
                    st.session_state["current_step"] = "batch_threshold"
                    st.rerun()
            else:
                st.warning("Close at least one polygon (>= 3 points) first.")


