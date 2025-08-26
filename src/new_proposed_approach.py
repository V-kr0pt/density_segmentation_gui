# -*- coding: utf-8 -*-
"""
DBT breast density segmentation pipeline:
- Load NIfTI volume
- Preprocess (normalize + CLAHE)
- Manual polygon on central slice -> manual threshold -> seed mask
- Convert seed to robust prompts (box + points)
- SAM2 video predictor on ORIGINAL slices (center-outward via 2 passes)
- RevSAM2: re-inject good masks as prompts to fix drift
- Refinement: adaptive threshold inside SAM2 ROI
- 3D post-processing
Requirements:
    pip install nibabel opencv-python numpy scipy
Assumes SAM2 installed per your snippet and predictor already built.
"""

from __future__ import annotations
import os
import math
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
import nibabel as nib
import torch

# ---------- CONFIG ----------

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


# ---------- I/O & PREPROCESS ----------

def load_nifti_as_float(path: str) -> np.ndarray:
    vol = nib.load(path).get_fdata(dtype=np.float32)  # (H, W, Z) or (X, Y, Z)
    if vol.ndim != 3:
        raise ValueError("Expected 3D NIfTI (H, W, Z).")
    return vol

def percentile_clip_norm(vol: np.ndarray, pct: float) -> np.ndarray:
    lo, hi = np.percentile(vol, [pct, 100 - pct])
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo + 1e-6)
    return vol.astype(np.float32)

def apply_clahe_per_slice(vol01: np.ndarray, cfg: PreprocCfg) -> np.ndarray:
    H, W, Z = vol01.shape
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_tile, cfg.clahe_tile))
    out = np.empty_like(vol01)
    for z in range(Z):
        sl = (vol01[..., z] * 255.0).astype(np.uint8)
        out[..., z] = clahe.apply(sl).astype(np.float32) / 255.0
    return out

def to_rgb_frames(vol01: np.ndarray, to_uint8: bool) -> List[np.ndarray]:
    frames = []
    for z in range(vol01.shape[-1]):
        g = vol01[..., z]
        if to_uint8:
            g8 = (np.clip(g, 0, 1) * 255.0).astype(np.uint8)
            rgb = np.stack([g8, g8, g8], axis=-1)
        else:
            rgb = np.stack([g, g, g], axis=-1).astype(np.float32)
        frames.append(rgb)
    return frames


# ---------- MANUAL ROI & SEED THRESHOLD ----------

def polygon_to_mask(poly_xy: np.ndarray, h: int, w: int) -> np.ndarray:
    """poly_xy: (N,2) in (x,y). Returns boolean mask."""
    poly = cv2.approxPolyDP(poly_xy.astype(np.float32), epsilon=1.5, closed=True)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    return mask.astype(bool)

def manual_threshold_in_roi(slice_img01: np.ndarray, roi_mask: np.ndarray, cfg: ThCfg) -> np.ndarray:
    """Threshold within ROI on normalized [0,1] image."""
    if slice_img01.ndim != 2:
        raise ValueError("manual_threshold_in_roi expects 2D slice (H,W).")
    thr_val = cfg.manual_threshold
    seed = (slice_img01 >= thr_val) & roi_mask

    # remove tiny components
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(seed, dtype=bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= cfg.min_comp_area_px:
            keep |= (lbl == i)
    return keep

def mask_to_box_and_points(mask: np.ndarray, max_points: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (box_xyxy float32), (points Nx2), (labels Nx1) for positives."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError("Seed mask empty.")
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    box = np.array([x0, y0, x1, y1], dtype=np.float32)

    # pick K positive points (centroids of largest components or grid within mask)
    num, lbl, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)]
    areas.sort(key=lambda t: t[1], reverse=True)
    pts = []
    for i, _ in areas[:max_points]:
        cx, cy = cents[i]  # (x, y)
        pts.append([cx, cy])
    if not pts:
        # fallback to center of box
        pts = [[(x0 + x1) / 2.0, (y0 + y1) / 2.0]]
    points = np.array(pts, dtype=np.float32)
    labels = np.ones((points.shape[0],), dtype=np.int32)  # positive points
    return box, points, labels


# ---------- SAM2 WRAPPERS (center-outward + RevSAM2) ----------

class Sam2DBTSegmenter:
    """
    Wraps SAM2VideoPredictor API:
      - init_state(frames)
      - add_new_points_or_box(state, ...)
      - propagate_in_video(state)
    See official README for API naming.  # ref: https://github.com/facebookresearch/sam2  (used via citation in the answer)
    """
    def __init__(self, predictor, sam_cfg: Sam2Cfg):
        self.predictor = predictor
        self.cfg = sam_cfg

    @torch.inference_mode()
    def _init_state(self, frames: List[np.ndarray]):
        # The predictor accepts a "video" abstraction. We pass a list of RGB frames.
        return self.predictor.init_state(frames)

    @torch.inference_mode()
    def _add_prompt(
        self,
        state,
        frame_idx: int,
        box_xyxy: Optional[np.ndarray],
        points_xy: Optional[np.ndarray],
        labels: Optional[np.ndarray],
        obj_id: int,
    ):
        # add_new_points_or_box(inference_state, frame_idx, obj_id, points, labels, box=...)
        return self.predictor.add_new_points_or_box(
            state,
            frame_idx,
            obj_id,
            points=points_xy,
            labels=labels,
            box=box_xyxy,
        )

    @torch.inference_mode()
    def _propagate(self, state) -> Dict[int, Dict[int, np.ndarray]]:
        """Returns masks[frame_idx][obj_id] = bool(H,W)."""
        out: Dict[int, Dict[int, np.ndarray]] = {}
        for out_fidx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(state):
            # out_mask_logits: (num_objs, 1, H, W) or (num_objs, H, W) depending on build
            logits = out_mask_logits
            if logits.ndim == 4:
                logits = logits[:, 0]
            pred = (logits > 0.0).detach().cpu().numpy().astype(np.uint8)
            out[out_fidx] = {int(oid): pred[i] for i, oid in enumerate(out_obj_ids)}
        return out

    def center_outward(
        self,
        frames: List[np.ndarray],
        center_idx: int,
        box: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        obj_id: int,
    ) -> Dict[int, np.ndarray]:
        """
        Two passes for stability:
          - Pass A: full propagation from center prompt (keeps all)
          - RevSAM2: iterate frames; good masks are re-injected as fresh prompts and re-propagated locally
        Returns a dense dict of per-slice masks (boolean).
        """
        H, W, _ = frames[0].shape
        # First global pass
        state = self._init_state(frames)
        self._add_prompt(state, center_idx, box, points, labels, obj_id)
        out_all = self._propagate(state)  # masks for many frames

        # Build initial sequence masks
        masks: Dict[int, np.ndarray] = {}
        prev_area = None
        prev_idx = None

        # Determine an ordering: center -> +1, +2, ... ; then center-1, center-2, ...
        order = [center_idx]
        for d in range(1, max(center_idx + 1, len(frames) - center_idx)):
            if center_idx + d < len(frames):
                order.append(center_idx + d)
            if center_idx - d >= 0:
                order.append(center_idx - d)

        # Iterate with RevSAM2 strategy
        for z in order:
            pred = out_all.get(z, {}).get(obj_id, None)
            if pred is None:
                continue
            area = int(pred.sum())
            ok = True
            if prev_area is not None and prev_area > 0:
                # area change constraint
                if abs(area - prev_area) / float(prev_area) > self.cfg.area_change_max:
                    ok = False

                # soft IoU with previous (if adjacent slices)
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
                # Re-inject: use this frame's current pred (even if shaky) as a box prompt and re-propagate locally
                ys, xs = np.where(pred > 0)
                if xs.size == 0 or ys.size == 0:
                    continue
                x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
                box_z = np.array([x0, y0, x1, y1], dtype=np.float32)
                # small local state to avoid overwriting whole memory bank
                local_state = self._init_state(frames)
                self._add_prompt(local_state, z, box_z, None, None, obj_id)
                local_out = self._propagate(local_state)
                pred2 = local_out.get(z, {}).get(obj_id, pred)
                masks[z] = (pred2 > 0).astype(bool)
                prev_area = int(masks[z].sum())
                prev_idx = z

        return masks


# ---------- REFINEMENT & POST ----------

def refine_within_roi(slice_img01: np.ndarray, roi_mask: np.ndarray, cfg: RefinementCfg) -> np.ndarray:
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

    # morphology
    if cfg.morph_open > 0:
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_open, cfg.morph_open))
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, k1).astype(bool)
    if cfg.morph_close > 0:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_close, cfg.morph_close))
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, k2).astype(bool)

    return out

def keep_largest_3d_component(mask3d: np.ndarray, min_voxels: int) -> np.ndarray:
    """mask3d: (H,W,Z) boolean."""
    vol = mask3d.astype(np.uint8)
    # 26-connectivity in 3D via scipy.ndimage if available; else simple OpenCV per-slice fallback
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
        # fallback: keep per-slice largest
        out = np.zeros_like(vol, dtype=np.uint8)
        H, W, Z = vol.shape
        for z in range(Z):
            n, lz, stats, _ = cv2.connectedComponentsWithStats(vol[..., z], connectivity=8)
            if n <= 1: 
                continue
            i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            out[..., z] = (lz == i).astype(np.uint8)
        return out.astype(bool)


# ---------- MAIN PIPELINE ----------

def run_pipeline(
    nii_path: str,
    polygon_xy_on_center: np.ndarray,
    predictor,  # from build_sam2_video_predictor(...)
    cfg: PipelineCfg,
) -> Dict[str, np.ndarray]:
    """
    Returns dict with:
        'sam2_roi_3d'   : bool (H,W,Z) propagated ROI from SAM2
        'density_3d'    : bool (H,W,Z) refined density mask
    """
    # 1) Load & preprocess
    vol = load_nifti_as_float(nii_path)             # (H,W,Z)
    vol01 = percentile_clip_norm(vol, cfg.pre.clip_percent)
    vol01 = apply_clahe_per_slice(vol01, cfg.pre)
    H, W, Z = vol01.shape

    # 2) Central slice & seed
    cidx = cfg.central_index if cfg.central_index is not None else Z // 2
    if not (0 <= cidx < Z):
        raise ValueError("central_index out of bounds.")
    roi_mask = polygon_to_mask(polygon_xy_on_center, H, W)
    seed = manual_threshold_in_roi(vol01[..., cidx], roi_mask, cfg.th)
    if seed.sum() == 0:
        raise RuntimeError("Seed after manual threshold is empty; adjust threshold or polygon.")

    # 3) Seed -> prompts
    box, pts, labs = mask_to_box_and_points(seed, max_points=5)

    # 4) Build frames (ORIGINAL slices, RGB)
    frames = to_rgb_frames(vol01, cfg.pre.out_uint8)

    # 5) SAM2 center-outward + RevSAM2
    segm = Sam2DBTSegmenter(predictor, cfg.sam)
    masks_sam: Dict[int, np.ndarray] = segm.center_outward(
        frames=frames,
        center_idx=cidx,
        box=box,
        points=pts,
        labels=labs,
        obj_id=cfg.sam.obj_id,
    )

    # 6) Fill missing slices by nearest neighbor (optional)
    # ensure every slice has a ROI (for refinement step)
    all_roi = np.zeros((H, W, Z), dtype=bool)
    known = sorted(masks_sam.keys())
    for z in range(Z):
        if z in masks_sam:
            all_roi[..., z] = masks_sam[z]
        else:
            # nearest known slice
            nn = min(known, key=lambda k: abs(k - z)) if known else cidx
            all_roi[..., z] = masks_sam.get(nn, np.zeros((H, W), bool))

    # 7) Refinement by adaptive threshold INSIDE ROI per slice
    density = np.zeros((H, W, Z), dtype=bool)
    for z in range(Z):
        if not all_roi[..., z].any():
            continue
        density[..., z] = refine_within_roi(vol01[..., z], all_roi[..., z], cfg.refine)

    # 8) Post-process 3D
    if cfg.post.keep_largest_3d:
        density = keep_largest_3d_component(density, cfg.post.min_3d_voxels)

    return {
        "sam2_roi_3d": all_roi,
        "density_3d": density,
    }


# ---------- EXAMPLE USAGE (fit to your existing predictor) ----------

if __name__ == "__main__":
    # You already built:
    # from sam2.build_sam import build_sam2_video_predictor
    # predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    # Configure
    pipeline_cfg = PipelineCfg(
        pre=PreprocCfg(clip_percent=0.5, clahe_clip=2.0, clahe_tile=8, out_uint8=True),
        th=ThCfg(manual_threshold=0.62, min_comp_area_px=64, poly_smooth_eps=1.5),
        sam=Sam2Cfg(obj_id=1, vos_optimized=False, iou_reinject_min=0.45, area_change_max=0.35),
        refine=RefinementCfg(method="otsu", morph_open=3, morph_close=3),
        post=PostprocCfg(keep_largest_3d=True, min_3d_voxels=500),
        central_index=None,  # default midpoint
    )

    # Inputs
    nii_path = "/path/to/your_dbt_volume.nii"
    # polygon in (x,y) for central slice; e.g., a list from your UI tool
    polygon_xy = np.array([
        [120, 200],
        [380, 190],
        [390, 420],
        [110, 430],
    ], dtype=np.float32)

    results = run_pipeline(
        nii_path=nii_path,
        polygon_xy_on_center=polygon_xy,
        predictor=predictor,  # from your setup
        cfg=pipeline_cfg,
    )

    # Save NIfTI masks aligned to input
    ref_img = nib.load(nii_path)
    affine = ref_img.affine
    nib.save(nib.Nifti1Image(results["sam2_roi_3d"].astype(np.uint8), affine), "sam2_roi_3d.nii.gz")
    nib.save(nib.Nifti1Image(results["density_3d"].astype(np.uint8), affine), "density_3d.nii.gz")
    print("Saved: sam2_roi_3d.nii.gz, density_3d.nii.gz")
