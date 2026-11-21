import numpy as np
import nibabel as nib
import re
import json
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
import pydicom
from pydicom.uid import generate_uid, ExplicitVRLittleEndian, RLELossless
from pydicom.dataset import FileMetaDataset


def apply_orientation(arr, rotate_k: int = 0, flip_h: bool = False, flip_v: bool = False, slice_dim: int | None = None):
    """
    Apply user-defined orientation (np.rot90 + flips) to a 2-D or 3-D array.
    For 3-D arrays the slice dimension is preserved; rotations and flips only
    touch the in-slice (row,col) axes.
    """
    arr = np.asarray(arr)
    rotate_k = int(rotate_k) % 4
    if arr.ndim == 2:
        if rotate_k != 0:
            arr = np.rot90(arr, k=rotate_k)
        if flip_h:
            arr = np.fliplr(arr)
        if flip_v:
            arr = np.flipud(arr)
        return np.ascontiguousarray(arr)
    if arr.ndim == 3:
        if slice_dim is None:
            slice_dim = int(np.argmin(arr.shape))
        other = [i for i in range(3) if i != slice_dim]
        if rotate_k != 0:
            arr = np.rot90(arr, k=rotate_k, axes=(other[0], other[1]))
        if flip_h:
            arr = np.flip(arr, axis=other[1])
        if flip_v:
            arr = np.flip(arr, axis=other[0])
        return arr
    raise ValueError(f"apply_orientation: unsupported ndim={arr.ndim}")


def orient_shape_3d(shape, slice_dim: int, rotate_k: int) -> tuple:
    """
    Return the new 3-D shape after np.rot90(k=rotate_k) around slice_dim.
    Flips don't change shape, so only rotation matters here.
    """
    rotate_k = int(rotate_k) % 4
    shape = list(shape)
    if rotate_k % 2 == 1:
        other = [i for i in range(3) if i != slice_dim]
        shape[other[0]], shape[other[1]] = shape[other[1]], shape[other[0]]
    return tuple(shape)


def sagittal_cw_rotate(volume: np.ndarray, affine: np.ndarray):
    """
    Rotate each 2-D slice 90° clockwise (axes 1 and 2 of a slices×rows×cols volume).

    The affine is intentionally NOT updated. The DICOM source has an identity affine
    (orientation tags missing). If we updated the affine correctly, an orientation-aware
    NIfTI viewer would undo the visual rotation by re-mapping physical coordinates —
    leaving the display unchanged. By keeping the original affine both original.nii.gz
    and mask.nii.gz share the same coordinate frame, so they stay registered with each
    other, and the viewer displays the rotated data directly without correction.
    """
    rotated = np.rot90(volume, k=3, axes=(1, 2))
    return rotated, affine


def resolve_dense_mask_path(output_path, prefer_dicom=False):
    """
    Resolve the saved dense mask path for either NIfTI or DICOM outputs.
    Returns a file path or folder path, or None if not found.
    """
    if prefer_dicom:
        candidates = [
            os.path.join(output_path, "dense_dcm"),
            os.path.join(output_path, "dense.dcm"),
            os.path.join(output_path, "dense.nii"),
        ]
    else:
        candidates = [
            os.path.join(output_path, "dense.nii"),
            os.path.join(output_path, "dense.dcm"),
            os.path.join(output_path, "dense_dcm"),
        ]
    for path in candidates:
        if os.path.isfile(path) or os.path.isdir(path):
            return path
    return None


def resolve_final_mask_path(output_path, prefer_dicom=False):
    """
    Resolve the saved final mask path for either NIfTI or DICOM outputs.
    Returns a file path or folder path, or None if not found.
    """
    # mask.nii.gz is always checked first (preferred output format)
    if prefer_dicom:
        candidates = [
            os.path.join(output_path, "dense_mask", "mask.nii.gz"),
            os.path.join(output_path, "dense_mask", "mask.nii"),
            os.path.join(output_path, "mask_dcm"),
            os.path.join(output_path, "mask.dcm"),
        ]
    else:
        candidates = [
            os.path.join(output_path, "dense_mask", "mask.nii.gz"),
            os.path.join(output_path, "dense_mask", "mask.nii"),
            os.path.join(output_path, "mask.dcm"),
            os.path.join(output_path, "mask_dcm"),
        ]
    for path in candidates:
        if os.path.isfile(path) or os.path.isdir(path):
            return path
    return None


def _ensure_2d_mask(mask):
    """Collapse a loaded mask to 2-D (squeeze, then pick central slice of any 3-D)."""
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim == 3:
        slice_dim = int(np.argmin(mask.shape))
        idx = mask.shape[slice_dim] // 2
        if slice_dim == 0:
            mask = mask[idx]
        elif slice_dim == 1:
            mask = mask[:, idx, :]
        else:
            mask = mask[:, :, idx]
    return mask


def save_polygon_mask_dcm(mask_2d, output_path, source_image_path=None):
    """
    Save the 2D polygon as a single-frame RLE-compressed DICOM at
    output_path/dense.dcm.

    Why this shape: a single 2D frame with RLE is typically tens of KB for
    binary polygons (huge zero-runs), writes almost instantly, and stays
    inspectable in any DICOM viewer. The old 3D-broadcast multi-frame
    approach produced ~5 MB files and spent real time encoding; the brief
    uncompressed .npy experiment was ~5 MB because raw binary masks don't
    shrink without compression.

    When source_image_path points at a DICOM (file or folder), patient /
    study tags are copied so the mask stays associated with its source
    series in viewers. For NIfTI sources minimal synthetic tags are used.

    Side effect: removes any transitional dense_mask_2d.npy in the same
    folder so there is only one source of truth going forward.
    """
    os.makedirs(output_path, exist_ok=True)
    mask_2d = np.asarray(mask_2d)
    if mask_2d.ndim > 2:
        mask_2d = np.squeeze(mask_2d)
    if mask_2d.ndim != 2:
        raise ValueError(
            f"save_polygon_mask_dcm expects a 2D mask, got shape {mask_2d.shape}"
        )
    mask_2d = np.ascontiguousarray((mask_2d > 0).astype(np.uint8) * 255)

    rows, cols = mask_2d.shape

    source_ds = None
    if source_image_path is not None:
        try:
            if os.path.isdir(source_image_path):
                from ImageLoader import DicomLoader
                files = DicomLoader._list_dicom_files(source_image_path)
                source_ds = pydicom.dcmread(files[0], stop_before_pixels=True)
            elif source_image_path.lower().endswith((".dcm", ".dicom")):
                source_ds = pydicom.dcmread(source_image_path, stop_before_pixels=True)
        except Exception:
            source_ds = None

    ds = pydicom.Dataset()
    if source_ds is not None:
        for tag in ("PatientID", "PatientName", "PatientBirthDate",
                    "StudyInstanceUID", "StudyDate", "StudyDescription",
                    "AccessionNumber"):
            if tag in source_ds:
                setattr(ds, tag, getattr(source_ds, tag))

    if "PatientID" not in ds:
        ds.PatientID = "ANON"
    if "StudyInstanceUID" not in ds:
        ds.StudyInstanceUID = generate_uid()

    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image
    ds.SOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "OT"
    ds.ImageType = ["DERIVED", "PRIMARY", "MASK"]
    ds.SeriesDescription = "dense polygon mask (2D)"
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = mask_2d.tobytes()

    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    try:
        ds.compress(RLELossless)
    except Exception as e:
        print(f"Warning: RLE compression failed ({e}); saving uncompressed.")

    out_path = os.path.join(output_path, "dense.dcm")
    ds.save_as(out_path, write_like_original=False)

    legacy_npy = os.path.join(output_path, "dense_mask_2d.npy")
    if os.path.isfile(legacy_npy):
        try:
            os.remove(legacy_npy)
        except Exception:
            pass

    return out_path


def load_polygon_mask_2d(output_path):
    """
    Load the saved 2D polygon mask, or None if nothing is saved.
    Handles, in order: the transitional .npy format, then any DICOM /
    NIfTI form (new single-frame RLE dense.dcm or legacy 3D). For 3-D
    sources the central slice is extracted — which is the polygon itself
    since the draw step replicated it across every slice.
    """
    npy_path = os.path.join(output_path, "dense_mask_2d.npy")
    if os.path.isfile(npy_path):
        return np.load(npy_path)

    mask_path = resolve_dense_mask_path(output_path)
    if mask_path is None:
        return None
    try:
        from ImageLoader import UnifiedImageLoader
        mask_slice, _, _, _, _ = UnifiedImageLoader.load_slice(mask_path)
        return _ensure_2d_mask(mask_slice)
    except Exception:
        return None


def has_polygon_mask(output_path):
    """Fast existence check for any saved polygon mask (new DCM, transitional .npy, or legacy 3-D)."""
    if os.path.isfile(os.path.join(output_path, "dense_mask_2d.npy")):
        return True
    return resolve_dense_mask_path(output_path) is not None


class ImageProcessor:
    """Image processing operations"""
    
    @staticmethod
    def normalize_image(img):
        """
        Return the image in 0-255 range for display.
        """
        img = img - np.min(img)
        img = img / (np.max(img) + 1e-8)
        return (img * 255).astype(np.uint8)
    
    @staticmethod
    def normalize_data(data):
        """
        Return data in 0-1 range.
        """
        mn, mx = data.min(), data.max()
        if mx == mn:
            return np.zeros_like(data)
        return (data - mn) / (mx - mn)


class DisplayTransform:
    """
    Tracks and applies/reverses display transformations.

    Principle:
      - Keep images in NATIVE orientation for all computations/saving.
      - Apply ONLY display transforms for GUI.
      - Always invert transforms for canvas->native coordinates.
    """

    def __init__(self, padding=50, rotate_k=0):
        # rotate_k: np.rot90 k-value applied for DISPLAY (0,1,2,3)
        self.padding = padding
        self.scale = 1.0
        self.padded_shape = None
        self.original_shape = None
        self.rotate_k = rotate_k  # display rotation

    def set_rotation_for_type(self, img_type: str):
        """
        Define display rotation by image type.

        For your case:
          - NIfTI should be displayed rotated -90° (clockwise) -> np.rot90(k=3)
          - DICOM displayed as-is -> k=0
        """
        self.rotate_k = 1 if img_type == "nii" else 0

    @staticmethod
    def _rot90_inverse_rc(rp, cp, H, W, k):
        """
        Inverse mapping of np.rot90 for indices.
        Given (rp,cp) in rotated image, return (r,c) in original image.
        """
        k = k % 4
        if k == 0:
            return rp, cp
        if k == 1:
            # rot90 CCW: (r,c)->(W-1-c, r)  => inverse: (rp,cp)->(cp, W-1-rp)
            return cp, (W - 1 - rp)
        if k == 2:
            # 180: (r,c)->(H-1-r, W-1-c) => inverse same
            return (H - 1 - rp), (W - 1 - cp)
        # k == 3 (CW): (r,c)->(c, H-1-r) => inverse: (rp,cp)->(H-1-cp, rp)
        return (H - 1 - cp), rp

    def prepare_for_display(self, image, img_type: str, max_width=1200, max_height=800):
        """
        Prepare image for canvas display:
          1) squeeze to 2D if a 3D array arrives (some DICOMs lack NumberOfFrames
             but pixel_array still returns (frames, rows, cols))
          2) pad
          3) normalize
          4) rotate (DISPLAY ONLY)
          5) scale to fit
        """

        # Guard: ensure we always work with a 2D slice
        if image.ndim == 3:
            slice_dim = int(np.argmin(image.shape))
            idx = image.shape[slice_dim] // 2
            if slice_dim == 0:
                image = image[idx]
            elif slice_dim == 1:
                image = image[:, idx, :]
            else:
                image = image[:, :, idx]

        self.original_shape = image.shape

        # 1) pad (native)
        padded = np.pad(
            image,
            pad_width=((self.padding, self.padding), (self.padding, self.padding)),
            mode="constant",
            constant_values=0,
        )
        self.padded_shape = padded.shape

        # 2) normalize (display)
        padded = ImageProcessor.normalize_image(padded)

        # 3) rotate for display (no effect on native computations)
        self.set_rotation_for_type(img_type)
        if self.rotate_k % 4 != 0:
            padded = np.rot90(padded, k=self.rotate_k)

        # 4) scale
        H, W = padded.shape
        self.scale = min(max_width / W, max_height / H, 1.0)

        pil_h = int(H * self.scale)
        pil_w = int(W * self.scale)
        return Image.fromarray(padded).resize((pil_w, pil_h))

    def canvas_to_native_coords(self, canvas_points):
        """
        Convert canvas (x,y) -> native (row,col).

        Inverse steps:
          1) undo scaling -> rotated+padded coords
          2) undo rotation -> padded coords
          3) undo padding -> native coords
        """
        native_points = []

        # Shape before rotation (padded native)
        Hp, Wp = self.padded_shape

        # Shape after rotation (display array)
        if self.rotate_k % 2 == 1:
            Hr, Wr = (Wp, Hp)  # swapped
        else:
            Hr, Wr = (Hp, Wp)

        for x, y in canvas_points:
            # 1) undo scaling: canvas pixels -> rotated array indices (row/col)
            rp = int(np.clip(y / self.scale, 0, Hr - 1))
            cp = int(np.clip(x / self.scale, 0, Wr - 1))

            # 2) undo rotation: rotated -> padded(native)
            r_pad, c_pad = self._rot90_inverse_rc(rp, cp, Hp, Wp, self.rotate_k)

            # 3) undo padding: padded -> native
            r = int(np.clip(r_pad - self.padding, 0, self.original_shape[0] - 1))
            c = int(np.clip(c_pad - self.padding, 0, self.original_shape[1] - 1))

            native_points.append((r, c))

        return native_points



class MaskManager:
    """Mask creation and saving operations with proper orientation handling."""
    
    @staticmethod
    def create_mask(image, points):
        """
        Creates a binary mask from a polygon defined by points.
        Points should be in the SAME coordinate system as the image.
        
        Args:
            image (numpy.ndarray): Input image (2D)
            points (list of tuples): Polygon coordinates as [(row1, col1), (row2, col2), ...]
                                    in NATIVE image coordinates
        
        Returns:
            tuple: (masked_image, binary_mask)
        """
        if len(points) < 3:
            raise ValueError("At least 3 points are required to create a polygon mask.")

        # Some loaders return a 3-D slice (e.g. 4-D NIfTI or odd DICOM layouts).
        # Mirror what DisplayTransform.prepare_for_display does: drop size-1
        # dims, and if still 3-D pick the middle index along the smallest axis.
        image = np.asarray(image)
        if image.ndim > 2:
            image = np.squeeze(image)
        if image.ndim == 3:
            slice_dim = int(np.argmin(image.shape))
            idx = image.shape[slice_dim] // 2
            if slice_dim == 0:
                image = image[idx]
            elif slice_dim == 1:
                image = image[:, idx, :]
            else:
                image = image[:, :, idx]
        if image.ndim != 2:
            raise ValueError(
                f"create_mask expects a 2D image, got shape {image.shape}"
            )

        mask = np.zeros(image.shape, dtype=np.uint8)

        # OpenCV expects (x, y) = (col, row)
        pts_xy = np.array([[(c, r) for (r, c) in points]], dtype=np.int32)

        mask = np.ascontiguousarray(mask)
        cv2.fillPoly(mask, pts_xy, 255)

        result = image * (mask > 0)
        return result, mask
    
    @staticmethod
    def create_combined_mask(image, polygons):
        """
        Creates a combined binary mask from multiple polygons.
        All polygons should be in NATIVE image coordinates.

        Args:
            image (numpy.ndarray): Input image (2D)
            polygons (list of lists): List of polygons, each as [(row, col), ...]

        Returns:
            tuple: (masked_image, combined_mask)
        """
        # Normalize to 2-D up front so combined_mask can't become 3-D via
        # broadcast when image comes in 3-D (4-D NIfTI, odd DICOMs, …).
        image = np.asarray(image)
        if image.ndim > 2:
            image = np.squeeze(image)
        if image.ndim == 3:
            slice_dim = int(np.argmin(image.shape))
            idx = image.shape[slice_dim] // 2
            if slice_dim == 0:
                image = image[idx]
            elif slice_dim == 1:
                image = image[:, idx, :]
            else:
                image = image[:, :, idx]

        combined_mask = np.zeros(image.shape, dtype=np.uint8)

        for poly in polygons:
            if len(poly) >= 3:
                _, mask = MaskManager.create_mask(image, poly)
                combined_mask = np.maximum(combined_mask, mask)  # Union

        result = image * (combined_mask > 0)
        return result, combined_mask
    
    @staticmethod
    def measure_mask_area(mask):
        """Return number of non-zero pixels in mask."""
        return np.count_nonzero(mask)
           

    @staticmethod
    def save_mask(
        mask_2d,
        original_shape,
        affine,
        file_path,
        img_type,
        file_name="dense.nii",
        original_image_path=None,
    ):
        """
        Saves a 2D mask as a 3D NIfTI file in NATIVE orientation.
        
        The key insight: The mask_2d is already in the NATIVE orientation of a slice.
        We just need to replicate it across all slices in the correct dimension.
        
        Args:
            mask_2d (numpy.ndarray): 2D binary mask in NATIVE slice orientation
            original_shape (tuple): 3D shape of the original volume (e.g., (512, 512, 50))
            affine (numpy.ndarray): 4x4 affine transformation matrix from original image
            file_path (str): Directory path where mask will be saved
            file_name (str): Output filename for NIfTI
            original_image_path (str): Original DICOM path (file or folder) for DICOM outputs
        
        Returns:
            str: Full path to saved mask file
        """
        if len(original_shape) != 3:
            raise ValueError(f"original_shape must be 3D, got {len(original_shape)}D")

        os.makedirs(file_path, exist_ok=True)

        # Force 2-D and a clean layout before any shape checks. Some upstream
        # paths can produce a (1,H,W) or non-contiguous mask and cv2.resize
        # fails silently in those cases.
        mask_2d = np.asarray(mask_2d)
        if mask_2d.ndim > 2:
            mask_2d = np.squeeze(mask_2d)
        if mask_2d.ndim != 2:
            raise ValueError(
                f"save_mask expects a 2D mask, got shape {mask_2d.shape}"
            )

        # Ensure binary values (0/255) for saved mask (better overlay visibility)
        mask_2d = np.ascontiguousarray((mask_2d > 0).astype(np.uint8) * 255)

        if img_type == "DICOM":
            print("Detected DICOM image!")
            if original_image_path is None:
                raise ValueError("original_image_path is required for DICOM mask export.")

        # Identify which dimension contains slices (usually smallest)
        slice_dim = int(np.argmin(original_shape))

        # Build the expected 2D shape for a slice
        other_dims = [i for i in range(3) if i != slice_dim]
        expected_slice_shape = (original_shape[other_dims[0]], original_shape[other_dims[1]])

        # Verify mask dimensions match expected slice dimensions.
        # Resize via PIL because cv2.resize can fail on very large arrays,
        # non-contiguous views, or dtype mismatches. PIL's NEAREST resampling
        # preserves binary values.
        if mask_2d.shape != expected_slice_shape:
            if mask_2d.shape == expected_slice_shape[::-1]:
                mask_2d = np.ascontiguousarray(mask_2d.T)
            else:
                print(
                    f"WARNING: Mask shape {mask_2d.shape} doesn't match expected "
                    f"{expected_slice_shape} (original_shape={original_shape}, "
                    f"slice_dim={slice_dim}). Resampling with PIL NEAREST."
                )
                # PIL size order is (width, height) = (cols, rows)
                target_pil_size = (expected_slice_shape[1], expected_slice_shape[0])
                mask_2d = np.array(
                    Image.fromarray(mask_2d).resize(target_pil_size, Image.NEAREST),
                    dtype=np.uint8,
                )
        
        # Replicate the 2D mask across the slice dimension in one C-level
        # copy via np.broadcast_to — much faster than a Python per-slice loop.
        slice_dim = int(np.argmin(original_shape))

        if slice_dim == 0:
            mask_view = np.broadcast_to(mask_2d[np.newaxis, :, :], original_shape)
        elif slice_dim == 1:
            mask_view = np.broadcast_to(mask_2d[:, np.newaxis, :], original_shape)
        else:
            mask_view = np.broadcast_to(mask_2d[:, :, np.newaxis], original_shape)

        mask_3d = np.ascontiguousarray(mask_view, dtype=np.uint8)

        if img_type == "DICOM":
            return MaskManager._save_dicom_mask(
                mask_3d=mask_3d,
                source_path=original_image_path,
                output_dir=file_path,
                base_name="dense",
            )

        # Save NIfTI with original affine - this preserves spatial registration
        mask_nii = nib.Nifti1Image(mask_3d, affine)
        full_path = os.path.join(file_path, file_name)
        nib.save(mask_nii, full_path)

        print(f"Mask saved to {full_path}")
        print(f"  Shape: {mask_3d.shape} (matches original: {original_shape})")
        print(f"  Slice dimension: {slice_dim}")

        return full_path

    @staticmethod
    def _compress_and_save(ds, out_path):
        """
        Save a DICOM dataset with RLE Lossless compression when possible.
        Binary masks are mostly long runs of zeros, so RLE typically shrinks
        the file ~50-100× with near-instant decode. Falls back to uncompressed
        if the encoder raises (older pydicom, unusual pixel layouts, etc.).
        """
        try:
            ds.compress(RLELossless)
        except Exception as e:
            print(f"Warning: RLE compression failed ({e}); saving uncompressed.")
        ds.save_as(out_path, write_like_original=False)

    @staticmethod
    def _update_dicom_for_mask(ds, mask_slice, series_uid, instance_number, series_desc):
        if getattr(ds, "file_meta", None) is None:
            ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = generate_uid()
        if getattr(ds, "file_meta", None) is None:
            ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.InstanceNumber = int(instance_number)
        ds.ImageType = ["DERIVED", "PRIMARY", "MASK"]
        ds.Modality = "OT"
        ds.SeriesDescription = series_desc
        ds.Rows, ds.Columns = mask_slice.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        if hasattr(ds, "NumberOfFrames"):
            del ds.NumberOfFrames
        if hasattr(ds, "RescaleIntercept"):
            ds.RescaleIntercept = 0
        if hasattr(ds, "RescaleSlope"):
            ds.RescaleSlope = 1
        ds.PixelData = mask_slice.astype(np.uint8).tobytes()
        return ds

    @staticmethod
    def _save_dicom_mask(mask_3d, source_path, output_dir, base_name):
        os.makedirs(output_dir, exist_ok=True)
        mask_3d = (mask_3d > 0).astype(np.uint8) * 255
        series_desc = f"{base_name} mask"

        if os.path.isdir(source_path):
            from ImageLoader import DicomLoader

            dicom_files = DicomLoader._list_dicom_files(source_path)
            meta_list = [(fp, DicomLoader._read_meta(fp)) for fp in dicom_files]
            meta_list = DicomLoader._sort_series(meta_list)

            num_slices = len(meta_list)

            # ---- Single-file multi-frame (e.g. DBT / tomosynthesis) ----
            # Check BEFORE creating out_dir so we don't leave an empty dense_dcm/
            # folder behind — resolve_dense_mask_path would find it and crash.
            if num_slices == 1 and int(getattr(meta_list[0][1], "NumberOfFrames", 1)) > 1:
                return MaskManager._save_dicom_mask(
                    mask_3d=mask_3d,
                    source_path=meta_list[0][0],
                    output_dir=output_dir,
                    base_name=base_name,
                )

            out_dir = os.path.join(output_dir, f"{base_name}_dcm")
            os.makedirs(out_dir, exist_ok=True)

            if mask_3d.shape[0] != num_slices:
                if mask_3d.shape[-1] == num_slices:
                    mask_3d = np.moveaxis(mask_3d, -1, 0)
                elif mask_3d.shape[1] == num_slices:
                    mask_3d = np.moveaxis(mask_3d, 1, 0)
                else:
                    raise ValueError(
                        f"Mask slices ({mask_3d.shape[0]}) do not match DICOM slices ({num_slices})."
                    )

            series_uid = generate_uid()
            for idx, (fp, _) in enumerate(meta_list):
                ds = pydicom.dcmread(fp)
                mask_slice = mask_3d[idx]
                if (ds.Rows, ds.Columns) != mask_slice.shape:
                    mask_slice = cv2.resize(
                        mask_slice, (int(ds.Columns), int(ds.Rows)), interpolation=cv2.INTER_NEAREST
                    )
                ds = MaskManager._update_dicom_for_mask(
                    ds, mask_slice, series_uid, idx + 1, series_desc
                )
                out_path = os.path.join(out_dir, f"IM_{idx + 1:04d}.dcm")
                MaskManager._compress_and_save(ds, out_path)

            print(f"DICOM mask series saved to {out_dir}")
            return out_dir

        ds = pydicom.dcmread(source_path)
        arr = ds.pixel_array
        series_uid = generate_uid()
        out_path = os.path.join(output_dir, f"{base_name}.dcm")

        if arr.ndim == 2:
            mask_slice = mask_3d[0]
            if (ds.Rows, ds.Columns) != mask_slice.shape:
                mask_slice = cv2.resize(
                    mask_slice, (int(ds.Columns), int(ds.Rows)), interpolation=cv2.INTER_NEAREST
                )
            ds = MaskManager._update_dicom_for_mask(
                ds, mask_slice, series_uid, 1, series_desc
            )
            MaskManager._compress_and_save(ds, out_path)
            print(f"DICOM mask saved to {out_path}")
            return out_path

        # Multi-frame single file
        mask_vol = mask_3d
        if mask_vol.ndim == 2:
            mask_vol = mask_vol[np.newaxis, ...]
        if mask_vol.shape[1] != ds.Rows or mask_vol.shape[2] != ds.Columns:
            resized = []
            for frame in mask_vol:
                resized.append(
                    cv2.resize(frame, (int(ds.Columns), int(ds.Rows)), interpolation=cv2.INTER_NEAREST)
                )
            mask_vol = np.stack(resized, axis=0)

        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = generate_uid()
        if getattr(ds, "file_meta", None) is None:
            ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.ImageType = ["DERIVED", "PRIMARY", "MASK"]
        ds.Modality = "OT"
        ds.SeriesDescription = series_desc
        ds.Rows, ds.Columns = mask_vol.shape[1], mask_vol.shape[2]
        ds.NumberOfFrames = mask_vol.shape[0]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        if hasattr(ds, "RescaleIntercept"):
            ds.RescaleIntercept = 0
        if hasattr(ds, "RescaleSlope"):
            ds.RescaleSlope = 1
        ds.PixelData = mask_vol.astype(np.uint8).tobytes()
        MaskManager._compress_and_save(ds, out_path)
        print(f"DICOM multi-frame mask saved to {out_path}")
        return out_path
    
    @staticmethod 
    def create_final_mask(folder_path, original_shape, original_affine, img_type="nii", original_image_path=None):
        """
        Creates a 3D NIfTI mask from processed slice arrays.
        
        Args:
            folder_path: directory containing .npy slice files
            original_shape: 3D shape of original volume
            original_affine: affine matrix from original volume
            
        Returns:
            str: path to saved mask.nii file
        """
        import re
        
        mask_path = os.path.join(folder_path, 'mask.nii')
        
        # Load all .npy files
        npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        if not npy_files:
            raise FileNotFoundError(f"No NPY files found in {folder_path}")
        
        # Sort files numerically by slice index to preserve correct order
        def extract_slice_index(filename):
            """Extract slice index from filename like 'slice_0005_threshold_0.38.npy'"""
            match = re.search(r'slice_(\d+)', filename)
            return int(match.group(1)) if match else 0
        
        npy_files.sort(key=extract_slice_index)
        
        print(f"Loading {len(npy_files)} slice arrays...")
        slices = []
        for f in npy_files:
            array = np.load(os.path.join(folder_path, f))
            # Convert to binary (0 or 1)
            binary_slice = (array > 0).astype(np.uint8) * 255
            slices.append(binary_slice)
        
        print(f"Original shape: {original_shape}")
        print(f"Slice shape: {slices[0].shape}")
        
        # Identify slice dimension
        slice_dim = np.argmin(original_shape)
        num_slices_expected = original_shape[slice_dim]
        
        if len(slices) != num_slices_expected:
            raise ValueError(
                f"Number of slice files ({len(slices)}) doesn't match "
                f"expected slices ({num_slices_expected}) for dimension {slice_dim}"
            )
        
        # Stack slices into 3D volume based on slice dimension
        other_dims = [i for i in range(3) if i != slice_dim]
        expected_slice_shape = (original_shape[other_dims[0]], original_shape[other_dims[1]])
        
        print(f"Expected slice shape: {expected_slice_shape}")
        
        # Verify and potentially transpose slices
        for i in range(len(slices)):
            if slices[i].shape != expected_slice_shape:
                if slices[i].shape == expected_slice_shape[::-1]:
                    slices[i] = slices[i].T
                else:
                    raise ValueError(
                        f"Slice {i} shape {slices[i].shape} doesn't match expected {expected_slice_shape}"
                    )
        
        # Stack along the correct dimension
        if slice_dim == 0:
            volume = np.stack(slices, axis=0)
        elif slice_dim == 1:
            volume = np.stack(slices, axis=1)
        else:  # slice_dim == 2
            volume = np.stack(slices, axis=2)
        
        print(f"Final volume shape: {volume.shape}")
        print(f"Expected shape: {original_shape}")
        
        if volume.shape != original_shape:
            raise ValueError(
                f"Final volume shape {volume.shape} doesn't match original {original_shape}"
            )
        
        # Rotate 90° CW in the sagittal plane so the output matches
        # the rotated original.nii.gz saved in batch_process_step.
        volume_out, affine_out = sagittal_cw_rotate(volume, original_affine)

        # Always save as .nii.gz for NIfTI viewer overlay compatibility
        mask_path = os.path.join(folder_path, 'mask.nii.gz')
        mask_nii = nib.Nifti1Image(volume_out.astype(np.uint8), affine_out)
        nib.save(mask_nii, mask_path)
        print(f"Final mask saved to {mask_path}")
        
        # Clean up temporary .npy files
        print(f"Cleaning up {len(npy_files)} temporary .npy files...")
        for f in npy_files:
            npy_path = os.path.join(folder_path, f)
            try:
                os.remove(npy_path)
            except Exception as e:
                print(f"Warning: Could not remove {f}: {e}")
        
        print(f"Cleanup complete. Temporary files removed.")
        
        return mask_path


class ThresholdOperator:
    """Threshold operations for segmentation."""
    
    @staticmethod
    def apply_threshold(image, mask, threshold):
        """Apply threshold to normalized image within mask region."""
        norm_image = ImageProcessor.normalize_data(image)
        return (norm_image > threshold) & (mask > 0)
    
    @staticmethod
    def threshold_slice(img, mask, threshold):
        """Apply threshold to a slice."""
        return ThresholdOperator.apply_threshold(img, mask, threshold)
    
    @staticmethod
    def display_thresholded_slice(img, mask, threshold):
        """
        Display thresholded result overlaid on image.
        
        Args:
            img: normalized image for display
            mask: binary mask
            threshold: threshold value
            
        Returns:
            matplotlib figure
        """
        bin_mask = ThresholdOperator.threshold_slice(img, mask, threshold)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img, cmap='gray')
        ax.imshow(bin_mask, cmap='jet', alpha=0.2)
        ax.text(0.5, 0.05, f'Threshold: {threshold:.2f}',
               ha='center', va='center',
               transform=ax.transAxes,
               color='white', fontsize=16)
        ax.axis('off')
        return fig
    
    @staticmethod
    def adjust_slice_threshold(image_slice, mask_slice, target_area):
        """
        Automatically adjusts threshold to match target area.
        
        Args:
            image_slice: image slice to threshold
            mask_slice: mask defining region of interest
            target_area: target number of pixels
            
        Returns:
            tuple: (best_threshold, thresholded_image)
        """
        threshold = 0.8
        step = 0.01
        best_threshold = threshold
        best_diff = float('inf')
        
        while threshold >= 0:
            thresholded_image = ThresholdOperator.threshold_slice(
                image_slice, mask_slice, threshold
            )
            area = MaskManager.measure_mask_area(thresholded_image)
            diff = abs(area - target_area)
            
            if diff < best_diff:
                best_diff = diff
                best_threshold = threshold
            
            threshold -= step
        
        final_thresholded = ThresholdOperator.threshold_slice(
            image_slice, mask_slice, best_threshold
        )
        return best_threshold, final_thresholded
