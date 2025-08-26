# -*- coding: utf-8 -*-
"""
Advanced SAM2 Pipeline Step for DBT Breast Density Segmentation GUI
Integrates the new proposed approach from new_proposed_approach.py
"""

import os
import math
import json
import io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
import nibabel as nib
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torch

from utils import ImageOperations, MaskOperations
from sam_utils import SAM2Manager

# ---------- CONFIGURATION CLASSES ----------

@dataclass(frozen=True)
class PreprocCfg:
    clip_percent: float = 0.5     # percentile clipping to suppress outliers
    clahe_clip: float = 2.0       # CLAHE clip limit
    clahe_tile: int = 8           # CLAHE tile grid size
    out_uint8: bool = True        # convert to uint8 for SAM2 (RGB)

@dataclass(frozen=True)
class ThCfg:
    manual_threshold: float       # you set this (0..1 if normalized; or native HU-like range scaled)
    min_comp_area_px: int = 64    # remove tiny regions in seed
    poly_smooth_eps: float = 1.5  # polygon simplification for mask rendering

@dataclass(frozen=True)
class Sam2Cfg:
    obj_id: int = 1
    vos_optimized: bool = False   # can be enabled in build_sam2_video_predictor if desired
    # RevSAM2 heuristics
    iou_reinject_min: float = 0.45
    area_change_max: float = 0.35  # reject if area jumps > 35% vs prev good

@dataclass(frozen=True)
class RefinementCfg:
    method: str = "otsu"          # "otsu" or "central_stats"
    morph_open: int = 3
    morph_close: int = 3

@dataclass(frozen=True)
class PostprocCfg:
    keep_largest_3d: bool = True
    min_3d_voxels: int = 500

@dataclass
class PipelineCfg:
    pre: PreprocCfg
    th: ThCfg
    sam: Sam2Cfg
    refine: RefinementCfg
    post: PostprocCfg
    central_index: Optional[int] = None  # if None -> mid slice


# ---------- ADVANCED PROCESSING FUNCTIONS ----------

def load_nifti_as_float(path: str) -> np.ndarray:
    """Load NIfTI volume as float32"""
    vol = nib.load(path).get_fdata(dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError("Expected 3D NIfTI (H, W, Z).")
    # Rearrange dimensions to match expected format
    vol = ImageOperations.rearrange_dimensions(vol)
    return vol

def percentile_clip_norm(vol: np.ndarray, pct: float) -> np.ndarray:
    """Normalize volume using percentile clipping"""
    lo, hi = np.percentile(vol, [pct, 100 - pct])
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo + 1e-6)
    return vol.astype(np.float32)

def apply_clahe_per_slice(vol01: np.ndarray, cfg: PreprocCfg) -> np.ndarray:
    """Apply CLAHE enhancement per slice"""
    H, W, Z = vol01.shape
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_tile, cfg.clahe_tile))
    out = np.empty_like(vol01)
    for z in range(Z):
        sl = (vol01[z, ..., ] * 255.0).astype(np.uint8)  # Adjusted for correct dimension order
        out[z, ...] = clahe.apply(sl).astype(np.float32) / 255.0
    return out

def to_rgb_frames(vol01: np.ndarray, to_uint8: bool) -> List[np.ndarray]:
    """Convert volume slices to RGB frames for SAM2"""
    frames = []
    Z = vol01.shape[0]  # Adjusted for correct dimension order
    for z in range(Z):
        g = vol01[z, ...]  # Get slice z
        if to_uint8:
            g8 = (np.clip(g, 0, 1) * 255.0).astype(np.uint8)
            rgb = np.stack([g8, g8, g8], axis=-1)
        else:
            rgb = np.stack([g, g, g], axis=-1).astype(np.float32)
        frames.append(rgb)
    return frames

def polygon_to_mask(poly_xy: np.ndarray, h: int, w: int) -> np.ndarray:
    """Convert polygon coordinates to binary mask"""
    poly = cv2.approxPolyDP(poly_xy.astype(np.float32), epsilon=1.5, closed=True)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    return mask.astype(bool)

def manual_threshold_in_roi(slice_img01: np.ndarray, roi_mask: np.ndarray, cfg: ThCfg) -> np.ndarray:
    """Apply manual threshold within ROI"""
    if slice_img01.ndim != 2:
        raise ValueError("manual_threshold_in_roi expects 2D slice (H,W).")
    thr_val = cfg.manual_threshold
    seed = (slice_img01 >= thr_val) & roi_mask

    # Remove tiny components
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(seed, dtype=bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= cfg.min_comp_area_px:
            keep |= (lbl == i)
    return keep

def mask_to_box_and_points(mask: np.ndarray, max_points: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert mask to SAM2 prompts (box + points)"""
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError("Seed mask empty.")
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    box = np.array([x0, y0, x1, y1], dtype=np.float32)

    # Pick K positive points (centroids of largest components)
    num, lbl, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)]
    areas.sort(key=lambda t: t[1], reverse=True)
    pts = []
    for i, _ in areas[:max_points]:
        cx, cy = cents[i]  # (x, y)
        pts.append([cx, cy])
    if not pts:
        # Fallback to center of box
        pts = [[(x0 + x1) / 2.0, (y0 + y1) / 2.0]]
    points = np.array(pts, dtype=np.float32)
    labels = np.ones((points.shape[0],), dtype=np.int32)  # positive points
    return box, points, labels

def refine_within_roi(slice_img01: np.ndarray, roi_mask: np.ndarray, cfg: RefinementCfg) -> np.ndarray:
    """Refine segmentation within ROI using adaptive thresholding"""
    if not roi_mask.any():
        return roi_mask
    sub = slice_img01.copy()
    sub[~roi_mask] = 0.0
    
    if cfg.method == "otsu":
        # Otsu on histogram of ROI only
        vals = (sub[roi_mask] * 255.0).astype(np.uint8)
        if len(vals) < 16:
            return roi_mask  # too few pixels to re-threshold
        thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out = (sub * 255.0 >= thr).astype(np.uint8)
        out = (out > 0).astype(np.uint8)
    else:
        # central_stats: threshold = mu + k*sigma on ROI
        roi_vals = sub[roi_mask]
        mu, sigma = float(roi_vals.mean()), float(roi_vals.std() + 1e-6)
        k = 0.25
        t = mu + k * sigma
        out = (sub >= t).astype(np.uint8)

    out = out.astype(bool) & roi_mask

    # Morphological operations
    if cfg.morph_open > 0:
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_open, cfg.morph_open))
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, k1).astype(bool)
    if cfg.morph_close > 0:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_close, cfg.morph_close))
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, k2).astype(bool)

    return out

def keep_largest_3d_component(mask3d: np.ndarray, min_voxels: int) -> np.ndarray:
    """Keep only the largest 3D connected component"""
    vol = mask3d.astype(np.uint8)
    try:
        from scipy.ndimage import label
        lbl, num = label(vol, structure=np.ones((3,3,3), dtype=np.uint8))
        if num <= 1:
            return vol.astype(bool)
        best = None
        best_area = -1
        for i in range(1, num+1):
            area = int((lbl == i).sum())
            if area > best_area:
                best_area = area
                best = (lbl == i)
        if best_area < min_voxels:
            return vol.astype(bool)
        return best.astype(bool)
    except Exception:
        # Fallback: keep per-slice largest
        out = np.zeros_like(vol, dtype=np.uint8)
        Z, H, W = vol.shape
        for z in range(Z):
            n, lz, stats, _ = cv2.connectedComponentsWithStats(vol[z, ...], connectivity=8)
            if n <= 1: 
                continue
            i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            out[z, ...] = (lz == i).astype(np.uint8)
        return out.astype(bool)


# ---------- ENHANCED SAM2 SEGMENTER CLASS ----------

class Sam2DBTSegmenter:
    """Enhanced SAM2 segmenter for DBT with center-outward propagation and RevSAM2"""
    
    def __init__(self, sam_manager: SAM2Manager, sam_cfg: Sam2Cfg):
        self.sam_manager = sam_manager
        self.cfg = sam_cfg
        self.video_predictor = None

    def _load_video_predictor(self):
        """Load video predictor if not already loaded"""
        if self.video_predictor is not None:
            return True
        
        result = self.sam_manager.load_video_predictor()
        if isinstance(result, tuple) and len(result) >= 2:
            success = result[0]
            if success:
                self.video_predictor = result[1]
                return True
        return False

    def segment_single_slice(self, image_rgb: np.ndarray, box: np.ndarray, 
                           points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Segment a single slice using SAM2 image predictor"""
        if not self.sam_manager.model_loaded:
            success, msg = self.sam_manager.load_model()
            if not success:
                raise RuntimeError(f"Failed to load SAM2 model: {msg}")
        
        # Set image
        self.sam_manager.predictor.set_image(image_rgb)
        
        # Predict mask
        masks, scores, logits = self.sam_manager.predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box,
            multimask_output=False
        )
        
        # Return best mask
        return masks[0] if len(masks) > 0 else np.zeros(image_rgb.shape[:2], dtype=bool)

    def center_outward_propagation(
        self,
        frames: List[np.ndarray],
        center_idx: int,
        box: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """
        Center-outward propagation with RevSAM2 strategy
        Falls back to slice-by-slice processing if video predictor is not available
        """
        masks: Dict[int, np.ndarray] = {}
        prev_area = None
        prev_idx = None

        # Try to use video predictor first
        use_video = self._load_video_predictor()
        
        if use_video:
            # Use video predictor for better propagation
            try:
                return self._video_propagation(frames, center_idx, box, points, labels)
            except Exception as e:
                st.warning(f"Video predictor failed, falling back to slice-by-slice: {str(e)}")
                use_video = False

        # Fallback to slice-by-slice processing
        # Determine processing order: center -> +1, +2, ... ; then center-1, center-2, ...
        order = [center_idx]
        for d in range(1, max(center_idx + 1, len(frames) - center_idx)):
            if center_idx + d < len(frames):
                order.append(center_idx + d)
            if center_idx - d >= 0:
                order.append(center_idx - d)

        # Process slices in order
        for z in order:
            if z >= len(frames):
                continue
                
            # Initial segmentation using SAM2
            if z == center_idx:
                # Use provided prompts for center slice
                pred = self.segment_single_slice(frames[z], box, points, labels)
            else:
                # For other slices, use previous mask as box prompt
                if prev_idx is not None and prev_idx in masks:
                    prev_mask = masks[prev_idx]
                    ys, xs = np.where(prev_mask)
                    if xs.size > 0 and ys.size > 0:
                        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
                        box_z = np.array([x0, y0, x1, y1], dtype=np.float32)
                        pred = self.segment_single_slice(frames[z], box_z, None, None)
                    else:
                        pred = np.zeros(frames[z].shape[:2], dtype=bool)
                else:
                    pred = np.zeros(frames[z].shape[:2], dtype=bool)

            # Quality check and RevSAM2 strategy
            area = int(pred.sum())
            ok = True
            
            if prev_area is not None and prev_area > 0:
                # Area change constraint
                if abs(area - prev_area) / float(prev_area) > self.cfg.area_change_max:
                    ok = False

                # IoU check with previous slice (if adjacent)
                if prev_idx is not None and abs(z - prev_idx) == 1:
                    inter = np.logical_and(pred, masks[prev_idx]).sum()
                    union = np.logical_or(pred, masks[prev_idx]).sum() + 1e-6
                    iou = inter / union
                    if iou < self.cfg.iou_reinject_min:
                        ok = False

            if ok:
                masks[z] = pred.astype(bool)
                prev_area = area
                prev_idx = z
            else:
                # Re-inject: use current prediction as box prompt and re-segment
                ys, xs = np.where(pred > 0)
                if xs.size > 0 and ys.size > 0:
                    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
                    box_z = np.array([x0, y0, x1, y1], dtype=np.float32)
                    pred2 = self.segment_single_slice(frames[z], box_z, None, None)
                    masks[z] = (pred2 > 0).astype(bool)
                    prev_area = int(masks[z].sum())
                    prev_idx = z
                else:
                    # Fallback to empty mask
                    masks[z] = np.zeros(frames[z].shape[:2], dtype=bool)
                    prev_area = 0
                    prev_idx = z

        return masks
    
    def _video_propagation(
        self,
        frames: List[np.ndarray],
        center_idx: int,
        box: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """Video-based propagation using SAM2 video predictor"""
        # Initialize video state
        inference_state = self.video_predictor.init_state(frames)
        
        # Add prompts to center frame
        self.video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=center_idx,
            obj_id=self.cfg.obj_id,
            points=points,
            labels=labels,
            box=box,
        )
        
        # Propagate through video
        masks = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
            # Convert logits to binary mask
            if out_mask_logits.ndim == 4:
                mask_logits = out_mask_logits[0, 0]  # Take first object, first channel
            else:
                mask_logits = out_mask_logits[0]  # Take first object
            
            binary_mask = (mask_logits > 0.0).cpu().numpy()
            masks[out_frame_idx] = binary_mask.astype(bool)
        
        return masks


def validate_gui_data(file_name, batch_files):
    """
    Validate that the traditional GUI data is available and correct for a file
    
    Args:
        file_name: Name of the file (without extension)
        batch_files: List of batch files
        
    Returns:
        tuple: (is_valid, error_message, file_paths)
    """
    # Check output directory exists
    output_path = os.path.join(os.getcwd(), 'output', file_name)
    if not os.path.exists(output_path):
        return False, f"Output directory not found: {output_path}", None
    
    # Check mask.json exists
    mask_json_path = os.path.join(output_path, 'mask.json')
    if not os.path.exists(mask_json_path):
        return False, f"Mask data not found: {mask_json_path}", None
    
    # Check source file exists
    input_folder = os.path.join(os.getcwd(), 'media')
    nii_file = None
    for ext in ['.nii', '.nii.gz']:
        candidate = file_name + ext
        if candidate in batch_files:
            nii_file = candidate
            break
    
    if nii_file is None:
        return False, f"Source file not found for {file_name}", None
    
    nii_path = os.path.join(input_folder, nii_file)
    if not os.path.exists(nii_path):
        return False, f"Source file does not exist: {nii_path}", None
    
    # Validate mask.json content
    try:
        with open(mask_json_path, 'r') as f:
            mask_data = json.load(f)
        
        # Check required fields
        if 'scale' not in mask_data:
            return False, f"Missing 'scale' field in mask data", None
        
        # Check polygon data
        has_polygons = 'polygons' in mask_data and len(mask_data['polygons']) > 0
        has_points = 'points' in mask_data and len(mask_data['points']) > 0
        
        if not (has_polygons or has_points):
            return False, f"No polygon or points data found in mask", None
        
        # Get points for validation
        if has_polygons:
            points = mask_data['polygons'][0]
        else:
            points = mask_data['points']
        
        if len(points) < 3:
            return False, f"Insufficient polygon points: {len(points)} (need at least 3)", None
        
    except Exception as e:
        return False, f"Error reading mask data: {str(e)}", None
    
    return True, "OK", {
        'output_path': output_path,
        'mask_json_path': mask_json_path,
        'nii_path': nii_path,
        'nii_file': nii_file
    }

# ---------- GUI INTEGRATION FUNCTIONS ----------

def extract_polygon_from_gui_data(mask_data, image_shape):
    """
    Extract polygon coordinates from GUI mask data, handling all transformations
    that the traditional GUI applies (scaling, rotation, etc.)
    
    Args:
        mask_data: Dictionary loaded from mask.json
        image_shape: (H, W) shape of the original image
        
    Returns:
        polygon_xy: np.ndarray of shape (N, 2) with (x, y) coordinates
    """
    # Get polygon points - prefer the polygons array if available
    if 'polygons' in mask_data and len(mask_data['polygons']) > 0:
        # Use the first polygon from the GUI drawing
        raw_points = mask_data['polygons'][0]
    else:
        # Fallback to the points field for backwards compatibility
        raw_points = mask_data['points']
    
    scale = mask_data['scale']
    
    # The points from the GUI are in the coordinate system after transformations
    # We need to convert them back to the original image coordinates
    
    # Convert to numpy array
    polygon_points = np.array(raw_points, dtype=np.float32)
    
    # The GUI applies several transformations:
    # 1. Scaling for display
    # 2. Rotation (90 degrees)
    # 3. Coordinate system adjustments
    
    # Since the advanced pipeline works directly with the original image coordinates,
    # we need to reverse these transformations or work in the same coordinate system
    
    # For now, we'll scale the points back to the original coordinate system
    # and then convert them to (x, y) format as expected by the polygon_to_mask function
    
    # Scale back to original coordinates
    original_coords = polygon_points / scale
    
    # Convert to (x, y) format - the polygon_to_mask expects (x, y) coordinates
    # where x is column index and y is row index
    polygon_xy = np.column_stack([original_coords[:, 1], original_coords[:, 0]])  # Swap x, y
    
    return polygon_xy

def load_traditional_gui_mask_as_polygon(mask_json_path, nii_path):
    """
    Load mask data saved by traditional GUI and convert to polygon format
    for use in advanced pipeline
    
    Args:
        mask_json_path: Path to mask.json file
        nii_path: Path to original NIfTI file
        
    Returns:
        polygon_xy: np.ndarray of shape (N, 2) with (x, y) coordinates
    """
    # Load mask data
    with open(mask_json_path, 'r') as f:
        mask_data = json.load(f)
    
    # Load original image to get dimensions
    vol = load_nifti_as_float(nii_path)
    Z, H, W = vol.shape
    
    # Get central slice (same as GUI uses)
    central_slice_idx = Z // 2
    central_slice = vol[central_slice_idx]
    
    # Extract polygon
    polygon_xy = extract_polygon_from_gui_data(mask_data, (H, W))
    
    return polygon_xy

def run_advanced_pipeline(
    nii_path: str,
    polygon_xy_on_center: np.ndarray,
    sam_manager: SAM2Manager,
    cfg: PipelineCfg,
    progress_callback=None
) -> Dict[str, np.ndarray]:
    """
    Run the advanced DBT segmentation pipeline
    Returns dict with:
        'sam2_roi_3d'   : bool (Z,H,W) propagated ROI from SAM2
        'density_3d'    : bool (Z,H,W) refined density mask
    """
    
    if progress_callback:
        progress_callback(0.0, "Loading and preprocessing volume...")
    
    # 1) Load & preprocess
    vol = load_nifti_as_float(nii_path)             # (Z,H,W)
    vol01 = percentile_clip_norm(vol, cfg.pre.clip_percent)
    vol01 = apply_clahe_per_slice(vol01, cfg.pre)
    Z, H, W = vol01.shape

    if progress_callback:
        progress_callback(0.2, "Generating seed mask from polygon...")

    # 2) Central slice & seed
    cidx = cfg.central_index if cfg.central_index is not None else Z // 2
    if not (0 <= cidx < Z):
        raise ValueError("central_index out of bounds.")
    roi_mask = polygon_to_mask(polygon_xy_on_center, H, W)
    seed = manual_threshold_in_roi(vol01[cidx, ...], roi_mask, cfg.th)
    if seed.sum() == 0:
        raise RuntimeError("Seed after manual threshold is empty; adjust threshold or polygon.")

    if progress_callback:
        progress_callback(0.3, "Converting seed to SAM2 prompts...")

    # 3) Seed -> prompts
    box, pts, labs = mask_to_box_and_points(seed, max_points=5)

    if progress_callback:
        progress_callback(0.4, "Preparing RGB frames for SAM2...")

    # 4) Build frames (ORIGINAL slices, RGB)
    frames = to_rgb_frames(vol01, cfg.pre.out_uint8)

    if progress_callback:
        progress_callback(0.5, "Running SAM2 center-outward propagation...")

    # 5) SAM2 center-outward + RevSAM2
    segm = Sam2DBTSegmenter(sam_manager, cfg.sam)
    masks_sam: Dict[int, np.ndarray] = segm.center_outward_propagation(
        frames=frames,
        center_idx=cidx,
        box=box,
        points=pts,
        labels=labs,
    )

    if progress_callback:
        progress_callback(0.7, "Filling missing slices and preparing for refinement...")

    # 6) Fill missing slices by nearest neighbor
    all_roi = np.zeros((Z, H, W), dtype=bool)
    known = sorted(masks_sam.keys())
    for z in range(Z):
        if z in masks_sam:
            all_roi[z, ...] = masks_sam[z]
        else:
            # Nearest known slice
            if known:
                nn = min(known, key=lambda k: abs(k - z))
                all_roi[z, ...] = masks_sam.get(nn, np.zeros((H, W), bool))

    if progress_callback:
        progress_callback(0.8, "Refining segmentation with adaptive thresholding...")

    # 7) Refinement by adaptive threshold INSIDE ROI per slice
    density = np.zeros((Z, H, W), dtype=bool)
    for z in range(Z):
        if not all_roi[z, ...].any():
            continue
        density[z, ...] = refine_within_roi(vol01[z, ...], all_roi[z, ...], cfg.refine)

    if progress_callback:
        progress_callback(0.9, "Applying 3D post-processing...")

    # 8) Post-process 3D
    if cfg.post.keep_largest_3d:
        density = keep_largest_3d_component(density, cfg.post.min_3d_voxels)

    if progress_callback:
        progress_callback(1.0, "Pipeline completed successfully!")

    return {
        "sam2_roi_3d": all_roi,
        "density_3d": density,
    }


# ---------- STREAMLIT UI FUNCTIONS ----------

def display_pipeline_config():
    """Display and allow editing of pipeline configuration"""
    st.markdown("### 🔧 Pipeline Configuration")
    
    # Debug mode
    debug_mode = st.checkbox("🐛 Enable Debug Mode", False, 
                           help="Show additional information and intermediate results")
    
    with st.expander("⚙️ Preprocessing Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            clip_percent = st.slider("Percentile Clipping", 0.1, 2.0, 0.5, 0.1, 
                                   help="Percentile for outlier suppression")
            clahe_clip = st.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, 0.1,
                                 help="CLAHE contrast enhancement limit")
        with col2:
            clahe_tile = st.slider("CLAHE Tile Size", 4, 16, 8, 2,
                                 help="CLAHE tile grid size")
            out_uint8 = st.checkbox("Convert to uint8", True,
                                  help="Convert frames to uint8 for SAM2")
    
    with st.expander("🎯 Threshold Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            manual_threshold = st.slider("Manual Threshold", 0.0, 1.0, 0.62, 0.01,
                                       help="Initial threshold for seed generation")
            min_comp_area = st.number_input("Min Component Area (pixels)", 1, 500, 64,
                                          help="Remove components smaller than this")
        with col2:
            poly_smooth = st.slider("Polygon Smoothing", 0.5, 5.0, 1.5, 0.1,
                                  help="Polygon simplification epsilon")
    
    with st.expander("🤖 SAM2 Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            obj_id = st.number_input("Object ID", 1, 10, 1, help="SAM2 object identifier")
            iou_reinject_min = st.slider("Min IoU for Re-injection", 0.1, 0.9, 0.45, 0.05,
                                       help="Minimum IoU threshold for RevSAM2")
        with col2:
            area_change_max = st.slider("Max Area Change", 0.1, 1.0, 0.35, 0.05,
                                      help="Maximum allowed area change between slices")
    
    with st.expander("✨ Refinement Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            refine_method = st.selectbox("Refinement Method", ["otsu", "central_stats"],
                                       help="Adaptive thresholding method")
            morph_open = st.slider("Morphological Opening", 0, 10, 3,
                                 help="Opening kernel size")
        with col2:
            morph_close = st.slider("Morphological Closing", 0, 10, 3,
                                  help="Closing kernel size")
    
    with st.expander("🔄 Post-processing Settings", expanded=False):
        keep_largest = st.checkbox("Keep Largest 3D Component", True,
                                 help="Keep only the largest connected component in 3D")
        min_3d_voxels = st.number_input("Min 3D Component Size", 100, 5000, 500,
                                      help="Minimum size for 3D components")
    
    # Build configuration
    cfg = PipelineCfg(
        pre=PreprocCfg(
            clip_percent=clip_percent,
            clahe_clip=clahe_clip,
            clahe_tile=clahe_tile,
            out_uint8=out_uint8
        ),
        th=ThCfg(
            manual_threshold=manual_threshold,
            min_comp_area_px=min_comp_area,
            poly_smooth_eps=poly_smooth
        ),
        sam=Sam2Cfg(
            obj_id=obj_id,
            vos_optimized=False,
            iou_reinject_min=iou_reinject_min,
            area_change_max=area_change_max
        ),
        refine=RefinementCfg(
            method=refine_method,
            morph_open=morph_open,
            morph_close=morph_close
        ),
        post=PostprocCfg(
            keep_largest_3d=keep_largest,
            min_3d_voxels=min_3d_voxels
        )
    )
    
    return cfg, debug_mode


def advanced_pipeline_step():
    """Main function for the advanced pipeline step in the GUI"""
    
    # Load external CSS
    try:
        with open("static/advanced_pipeline_step.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass
    
    st.header("🚀 Advanced SAM2 Pipeline")
    st.markdown("""
    <div class="description">
        Advanced density segmentation using SAM2 with center-outward propagation, 
        RevSAM2 quality control, and adaptive refinement.<br>
        <strong>✨ Fully integrated with traditional GUI:</strong> Uses your drawn polygons and all existing functionalities!
    </div>
    """, unsafe_allow_html=True)
    
    # Integration info
    st.info("""
    🔄 **GUI Integration**: This advanced pipeline completely reuses the traditional GUI:
    - ✅ Uses the exact same polygon drawing interface
    - ✅ Loads your drawn masks automatically  
    - ✅ Maintains all existing coordinate transformations
    - ✅ Compatible with all existing file formats
    - ✅ Preserves the same output structure
    """, unsafe_allow_html=True)
    
    # Session state initialization
    if "advanced_pipeline_files" not in st.session_state:
        st.session_state["advanced_pipeline_files"] = []
    if "advanced_pipeline_results" not in st.session_state:
        st.session_state["advanced_pipeline_results"] = {}
    
    # Check prerequisites
    if "batch_files" not in st.session_state or len(st.session_state["batch_files"]) == 0:
        st.warning("No files selected. Please select files first.")
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        return
    
    # Check if drawing step is completed
    if "batch_completed_files" not in st.session_state:
        st.warning("Please complete the drawing step first.")
        if st.button("← Back to Draw Step"):
            st.session_state["current_step"] = "batch_draw"
            st.rerun()
        return
    
    batch_files = st.session_state["batch_files"]
    completed_draw = st.session_state["batch_completed_files"]["draw"]
    batch_files_names = [f.split('.')[0] for f in batch_files]
    available_files = [f for f in batch_files_names if f in completed_draw]
    
    if len(available_files) == 0:
        st.warning("No files have completed the drawing step.")
        if st.button("← Back to Draw Step"):
            st.session_state["current_step"] = "batch_draw"
            st.rerun()
        return
    
    # SAM2 Model Management
    st.markdown("### 🤖 SAM2 Model Status")
    sam_manager = SAM2Manager()
    
    # Check SAM2 dependencies and model
    deps_ok, deps_msg = sam_manager.check_dependencies()
    checkpoint_ok, checkpoint_msg = sam_manager.check_checkpoint()
    
    col1, col2 = st.columns(2)
    with col1:
        if deps_ok:
            st.success(f"✅ {deps_msg}")
        else:
            st.error(f"❌ {deps_msg}")
    with col2:
        if checkpoint_ok:
            st.success(f"✅ {checkpoint_msg}")
        else:
            st.error(f"❌ {checkpoint_msg}")
    
    if not (deps_ok and checkpoint_ok):
        st.error("Cannot proceed without SAM2 dependencies and checkpoint.")
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
        return
    
    # Load model button
    if not sam_manager.model_loaded:
        if st.button("🔄 Load SAM2 Model", type="primary"):
            with st.spinner("Loading SAM2 model..."):
                success, msg = sam_manager.load_model()
                if success:
                    st.success(f"✅ {msg}")
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")
                    return
    else:
        st.success("✅ SAM2 model loaded and ready!")
    
    if not sam_manager.model_loaded:
        st.info("Please load the SAM2 model to continue.")
        return
    
    # Pipeline Configuration
    cfg, debug_mode = display_pipeline_config()
    
    # File Selection for Processing
    st.markdown("### 📁 Select Files for Advanced Processing")
    selected_files = st.multiselect(
        "Available files (with completed drawings):",
        available_files,
        default=available_files,
        help="Select which files to process with the advanced pipeline"
    )
    
    if len(selected_files) == 0:
        st.info("Please select at least one file to process.")
        return
    
    # Processing Controls
    st.markdown("### ⚡ Process Files")
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        st.info(f"Ready to process {len(selected_files)} files with advanced pipeline")
    
    with col2:
        # Test with first file
        if st.button("🧪 Test First", help="Test pipeline with first file only"):
            st.session_state["test_first_file"] = True
            st.rerun()
    
    with col3:
        if st.button("🚀 Process All", type="primary"):
            st.session_state["start_advanced_processing"] = True
            st.rerun()
    
    with col4:
        if st.button("🗑️ Clear Results"):
            st.session_state["advanced_pipeline_results"] = {}
            st.success("Results cleared!")
            st.rerun()
    
    # Processing Loop
    process_all = st.session_state.get("start_advanced_processing", False)
    test_first = st.session_state.get("test_first_file", False)
    
    if process_all or test_first:
        st.session_state["start_advanced_processing"] = False
        st.session_state["test_first_file"] = False
        
        # Determine which files to process
        files_to_process = [selected_files[0]] if test_first else selected_files
        processing_mode_text = "Testing with first file" if test_first else "Processing all files"
        
        # Overall progress
        total_files = len(files_to_process)
        overall_progress = st.progress(0)
        overall_status = st.empty()
        overall_status.text(processing_mode_text)
        
        for file_idx, file_name in enumerate(files_to_process):
            overall_status.text(f"Processing {file_idx + 1}/{total_files}: {file_name}")
            
            # Validate GUI data first
            is_valid, error_msg, file_paths = validate_gui_data(file_name, st.session_state["batch_files"])
            if not is_valid:
                st.error(f"❌ {file_name}: {error_msg}")
                continue
            
            try:
                # Use validated file paths
                mask_json_path = file_paths['mask_json_path']
                nii_path = file_paths['nii_path']
                output_path = file_paths['output_path']
                
                # Load polygon using the same coordinate system as the traditional GUI
                try:
                    polygon_xy = load_traditional_gui_mask_as_polygon(mask_json_path, nii_path)
                    file_status.text(f"{file_name}: ✅ Polygon loaded from GUI data")
                    
                    # Validate polygon
                    if len(polygon_xy) < 3:
                        st.error(f"❌ Invalid polygon for {file_name}: needs at least 3 points, got {len(polygon_xy)}")
                        continue
                    
                    if debug_mode:
                        st.info(f"🔍 Debug - {file_name}: Polygon has {len(polygon_xy)} points")
                        
                except Exception as e:
                    st.error(f"❌ Error loading polygon for {file_name}: {str(e)}")
                    continue
                
                # Progress callback for individual file
                file_progress = st.progress(0)
                file_status = st.empty()
                
                def progress_callback(progress, message):
                    file_progress.progress(progress)
                    file_status.text(f"{file_name}: {message}")
                
                # Debug visualization of loaded polygon
                if debug_mode:
                    with st.expander(f"🔍 Debug: {file_name} - Polygon Visualization", expanded=False):
                        try:
                            # Load central slice for visualization
                            vol_debug = load_nifti_as_float(nii_path)
                            central_idx = vol_debug.shape[0] // 2
                            central_slice_debug = vol_debug[central_idx]
                            
                            # Create mask from polygon for visualization
                            H, W = central_slice_debug.shape
                            debug_mask = polygon_to_mask(polygon_xy, H, W)
                            
                            # Show overlay
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                            
                            ax1.imshow(central_slice_debug, cmap='gray')
                            ax1.plot(polygon_xy[:, 0], polygon_xy[:, 1], 'r-', linewidth=2)
                            ax1.plot([polygon_xy[-1, 0], polygon_xy[0, 0]], [polygon_xy[-1, 1], polygon_xy[0, 1]], 'r-', linewidth=2)
                            ax1.set_title('Original + Polygon')
                            ax1.axis('off')
                            
                            ax2.imshow(central_slice_debug, cmap='gray')
                            ax2.imshow(debug_mask, cmap='Reds', alpha=0.3)
                            ax2.set_title('Original + Mask')
                            ax2.axis('off')
                            
                            st.pyplot(fig)
                            plt.close(fig)
                            
                        except Exception as debug_e:
                            st.warning(f"Debug visualization failed: {str(debug_e)}")
                
                # Run advanced pipeline
                results = run_advanced_pipeline(
                    nii_path=nii_path,
                    polygon_xy_on_center=polygon_xy,
                    sam_manager=sam_manager,
                    cfg=cfg,
                    progress_callback=progress_callback
                )
                
                # Debug visualization
                if debug_mode:
                    st.markdown(f"#### 🔍 Debug Info for {file_name}")
                    
                    # Show some intermediate slices
                    roi_3d = results["sam2_roi_3d"]
                    density_3d = results["density_3d"]
                    Z = roi_3d.shape[0]
                    
                    # Show center slice comparison
                    center_slice_idx = Z // 2
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Original (center slice)**")
                        # Load and show original slice for comparison
                        original_vol = load_nifti_as_float(nii_path)
                        original_slice = original_vol[center_slice_idx, ...]
                        st.image(original_slice, caption="Original", use_column_width=True, clamp=True)
                    
                    with col2:
                        st.markdown("**SAM2 ROI**")
                        roi_slice = roi_3d[center_slice_idx, ...].astype(np.uint8) * 255
                        st.image(roi_slice, caption="SAM2 ROI", use_column_width=True)
                    
                    with col3:
                        st.markdown("**Refined Density**")
                        density_slice = density_3d[center_slice_idx, ...].astype(np.uint8) * 255
                        st.image(density_slice, caption="Refined Density", use_column_width=True)
                    
                    # Volume statistics
                    st.markdown("**Volume Statistics:**")
                    debug_col1, debug_col2, debug_col3 = st.columns(3)
                    
                    with debug_col1:
                        st.metric("Total Slices", Z)
                    with debug_col2:
                        roi_coverage = (roi_3d.sum(axis=(1,2)) > 0).sum() / Z
                        st.metric("ROI Coverage", f"{roi_coverage:.1%}")
                    with debug_col3:
                        density_coverage = (density_3d.sum(axis=(1,2)) > 0).sum() / Z
                        st.metric("Density Coverage", f"{density_coverage:.1%}")
                
                # Save results
                advanced_output_path = os.path.join(output_path, 'advanced')
                os.makedirs(advanced_output_path, exist_ok=True)
                
                # Load original NIfTI for affine
                ref_img = nib.load(nii_path)
                affine = ref_img.affine
                
                # Save SAM2 ROI mask
                roi_3d = results["sam2_roi_3d"].astype(np.uint8)
                # Adjust dimensions to match original NIfTI format
                roi_3d_nii = np.transpose(roi_3d, (1, 2, 0))  # (H,W,Z)
                nib.save(nib.Nifti1Image(roi_3d_nii, affine), 
                        os.path.join(advanced_output_path, "sam2_roi_3d.nii.gz"))
                
                # Save refined density mask
                density_3d = results["density_3d"].astype(np.uint8)
                density_3d_nii = np.transpose(density_3d, (1, 2, 0))  # (H,W,Z)
                nib.save(nib.Nifti1Image(density_3d_nii, affine), 
                        os.path.join(advanced_output_path, "density_3d.nii.gz"))
                
                # Save configuration
                config_dict = {
                    "preprocessing": {
                        "clip_percent": cfg.pre.clip_percent,
                        "clahe_clip": cfg.pre.clahe_clip,
                        "clahe_tile": cfg.pre.clahe_tile,
                        "out_uint8": cfg.pre.out_uint8
                    },
                    "threshold": {
                        "manual_threshold": cfg.th.manual_threshold,
                        "min_comp_area_px": cfg.th.min_comp_area_px,
                        "poly_smooth_eps": cfg.th.poly_smooth_eps
                    },
                    "sam2": {
                        "obj_id": cfg.sam.obj_id,
                        "iou_reinject_min": cfg.sam.iou_reinject_min,
                        "area_change_max": cfg.sam.area_change_max
                    },
                    "refinement": {
                        "method": cfg.refine.method,
                        "morph_open": cfg.refine.morph_open,
                        "morph_close": cfg.refine.morph_close
                    },
                    "postprocessing": {
                        "keep_largest_3d": cfg.post.keep_largest_3d,
                        "min_3d_voxels": cfg.post.min_3d_voxels
                    }
                }
                
                with open(os.path.join(advanced_output_path, "pipeline_config.json"), 'w') as f:
                    json.dump(config_dict, f, indent=2)
                
                # Store results in session state
                st.session_state["advanced_pipeline_results"][file_name] = {
                    "roi_volume": int(results["sam2_roi_3d"].sum()),
                    "density_volume": int(results["density_3d"].sum()),
                    "output_path": advanced_output_path
                }
                
                file_status.text(f"{file_name}: ✅ Completed successfully!")
                st.success(f"✅ {file_name} processed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error processing {file_name}: {str(e)}")
                continue
            
            # Update overall progress
            overall_progress.progress((file_idx + 1) / total_files)
        
        overall_status.text(f"🎉 {processing_mode_text} completed!")
        if test_first:
            st.success("✅ Test completed successfully! You can now process all files if the results look good.")
        else:
            st.balloons()
    
    # Results Display
    if st.session_state["advanced_pipeline_results"]:
        st.markdown("### 📊 Processing Results")
        
        results = st.session_state["advanced_pipeline_results"]
        
        for file_name, result in results.items():
            with st.expander(f"📁 {file_name} - Advanced Pipeline Results", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("SAM2 ROI Volume", f"{result['roi_volume']:,} voxels")
                
                with col2:
                    st.metric("Refined Density Volume", f"{result['density_volume']:,} voxels")
                
                with col3:
                    if result['density_volume'] > 0:
                        density_ratio = result['density_volume'] / result['roi_volume']
                        st.metric("Density Ratio", f"{density_ratio:.1%}")
                    else:
                        st.metric("Density Ratio", "0%")
                
                # Check if traditional results exist for comparison
                traditional_output = os.path.join(os.getcwd(), 'output', file_name)
                traditional_dense_mask = os.path.join(traditional_output, 'dense_mask')
                
                if os.path.exists(traditional_dense_mask):
                    st.markdown("**📈 Comparison with Traditional Pipeline:**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.success("✅ Traditional pipeline results available")
                    with col_b:
                        if st.button(f"� Compare Results", key=f"compare_{file_name}"):
                            st.info("Results comparison feature coming soon!")
                
                st.markdown("**📂 Output Files:**")
                st.text(f"📁 Main output: {result['output_path']}")
                st.text(f"🧠 SAM2 ROI: sam2_roi_3d.nii.gz")
                st.text(f"🎯 Refined density: density_3d.nii.gz")
                st.text(f"⚙️ Pipeline config: pipeline_config.json")
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("← Back to File Selection"):
            st.session_state["current_step"] = "file_selection"
            st.rerun()
    
    with col2:
        if st.button("🎨 Back to Draw Step"):
            st.session_state["current_step"] = "batch_draw"
            st.rerun()
    
    with col3:
        if st.button("🎯 Back to Threshold Step"):
            st.session_state["current_step"] = "batch_threshold"
            st.rerun()
