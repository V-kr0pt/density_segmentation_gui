import os
import numpy as np
import pydicom
import nibabel as nib


class BaseImageLoader:
    """Base class for medical image loading operations."""

    @staticmethod
    def get_central_slice_index(shape):
        """
        Determines the central slice index for a 3D volume.
        The slice dimension is assumed to be the smallest dimension.
        """
        slice_dim = int(np.argmin(shape))
        return int(shape[slice_dim] // 2)


class NiftiLoader(BaseImageLoader):
    """Specialized loader for NIfTI files - loads in NATIVE orientation."""

    img_type = "nii"

    @staticmethod
    def load_volume(file_path: str, lazy: bool = False):
        """
        Loads NIfTI volume in its NATIVE orientation.
        Returns: (data, affine, original_shape, img_type)

        Uses float32 instead of the nibabel default float64 — halves peak
        memory for large tomosynthesis volumes with no precision loss for
        downstream normalisation / thresholding.
        """
        img = nib.load(file_path)
        if lazy:
            data = np.asarray(img.dataobj)  # memory-mapped when possible
        else:
            data = img.get_fdata(dtype=np.float32)

        original_shape = data.shape
        return data, img.affine, original_shape, NiftiLoader.img_type

    @staticmethod
    def load_slice(file_path: str, slice_index: int | None = None):
        """
        Loads a specific slice from a NIfTI file in NATIVE orientation.
        Returns: (slice_data, affine, original_shape, slice_index_used, img_type)
        """
        img = nib.load(file_path)
        original_shape = img.shape

        # 2D NIfTI
        if len(original_shape) == 2:
            slice_data = np.asarray(img.dataobj).astype(np.float32)
            return slice_data, img.affine, original_shape, 0, NiftiLoader.img_type

        # 3D+ : slice dimension assumed smallest
        slice_dim = int(np.argmin(original_shape))

        if slice_index is None:
            slice_index = int(original_shape[slice_dim] // 2)

        if slice_index < 0 or slice_index >= original_shape[slice_dim]:
            raise ValueError(
                f"Slice index {slice_index} out of range for dimension {slice_dim} "
                f"with size {original_shape[slice_dim]}"
            )

        # Efficient slice extraction via dataobj
        if slice_dim == 0:
            slice_data = np.asarray(img.dataobj[slice_index, :, :], dtype=np.float32)
        elif slice_dim == 1:
            slice_data = np.asarray(img.dataobj[:, slice_index, :], dtype=np.float32)
        else:
            slice_data = np.asarray(img.dataobj[:, :, slice_index], dtype=np.float32)

        return slice_data, img.affine, original_shape, slice_index, NiftiLoader.img_type


class DicomLoader(BaseImageLoader):
    """Specialized loader for DICOM series/files - loads in NATIVE orientation."""

    img_type = "DICOM"

    # -----------------------------
    # Helpers (fast + robust)
    # -----------------------------
    @staticmethod
    def _list_dicom_files(folder_path: str) -> list[str]:
        # English comment: fast directory scan; accepts .dcm/.dicom
        files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".dcm", ".dicom"))
        ]
        if not files:
            raise FileNotFoundError(f"No DICOM files found in {folder_path}")
        return files

    @staticmethod
    def _read_meta(path: str):
        # English comment: metadata-only read is much faster than loading PixelData
        return pydicom.dcmread(path, stop_before_pixels=True)

    @staticmethod
    def _safe_float_list(ds, attr: str, n: int | None = None):
        v = getattr(ds, attr, None)
        if v is None:
            return None
        try:
            arr = [float(x) for x in v]
        except Exception:
            return None
        if n is not None and len(arr) != n:
            return None
        return arr

    @staticmethod
    def _sort_series(meta_list: list[tuple[str, object]]) -> list[tuple[str, object]]:
        """
        Sort series by ImagePositionPatient projected on slice normal when possible.
        Fallback: InstanceNumber.
        """
        ds0 = meta_list[0][1]
        iop = DicomLoader._safe_float_list(ds0, "ImageOrientationPatient", 6)

        if iop is not None:
            row_cos = np.array(iop[:3], dtype=np.float32)
            col_cos = np.array(iop[3:], dtype=np.float32)
            slice_cos = np.cross(row_cos, col_cos).astype(np.float32)

            def key(item):
                ds = item[1]
                ipp = DicomLoader._safe_float_list(ds, "ImagePositionPatient", 3)
                if ipp is None:
                    return float(getattr(ds, "InstanceNumber", 0))
                return float(np.dot(np.array(ipp, dtype=np.float32), slice_cos))

            return sorted(meta_list, key=key)

        return sorted(meta_list, key=lambda x: int(getattr(x[1], "InstanceNumber", 0)))

    @staticmethod
    def _compute_affine_series(ds0, ds1=None) -> np.ndarray:
        """
        Compute NIfTI affine for a volume stacked as (k, r, c) = (slice, row, col).
        Uses:
          - ImageOrientationPatient (6)
          - ImagePositionPatient (3)
          - PixelSpacing (2)
          - slice spacing: from IPP delta projected onto normal (preferred),
            else SpacingBetweenSlices / SliceThickness.
        """
        iop = DicomLoader._safe_float_list(ds0, "ImageOrientationPatient", 6)
        ipp0 = DicomLoader._safe_float_list(ds0, "ImagePositionPatient", 3)
        ps = DicomLoader._safe_float_list(ds0, "PixelSpacing", 2)

        if iop is None or ipp0 is None or ps is None:
            import warnings
            warnings.warn(
                "Missing DICOM orientation tags; using identity affine.",
                UserWarning, stacklevel=2,
            )
            return np.eye(4, dtype=np.float32)

        row_cos = np.array(iop[:3], dtype=np.float32)
        col_cos = np.array(iop[3:], dtype=np.float32)
        slice_cos = np.cross(row_cos, col_cos).astype(np.float32)

        row_spacing = float(ps[0])
        col_spacing = float(ps[1])

        slice_spacing = None
        if ds1 is not None:
            ipp1 = DicomLoader._safe_float_list(ds1, "ImagePositionPatient", 3)
            if ipp1 is not None:
                delta = np.array(ipp1, dtype=np.float32) - np.array(ipp0, dtype=np.float32)
                slice_spacing = float(abs(np.dot(delta, slice_cos)))

        if slice_spacing is None or slice_spacing == 0.0:
            sbs = getattr(ds0, "SpacingBetweenSlices", None)
            thk = getattr(ds0, "SliceThickness", None)
            if sbs is not None:
                slice_spacing = float(sbs)
            elif thk is not None:
                slice_spacing = float(thk)
            else:
                slice_spacing = 1.0

        aff = np.eye(4, dtype=np.float32)
        # Indexing is (k, r, c) in LPS
        aff[:3, 0] = slice_cos * slice_spacing
        aff[:3, 1] = row_cos * row_spacing
        aff[:3, 2] = col_cos * col_spacing
        aff[:3, 3] = np.array(ipp0, dtype=np.float32)

        # Convert LPS (DICOM) -> RAS (NIfTI) so viewers align correctly.
        lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float32)
        return lps_to_ras @ aff

    @staticmethod
    def _normalize_pixel_array(arr: np.ndarray) -> np.ndarray:
        """
        Coerce a pydicom pixel_array to 3D (frames, rows, cols).
        Handles the shapes we've actually seen in the wild:
          - 2D (rows, cols)                       -> wrapped as (1, rows, cols)
          - 3D (frames, rows, cols)               -> passthrough
          - 4D (frames, rows, cols, 1)            -> squeeze trailing sample axis
        Some vendors/PACS round-trips encode monochrome DBT with
        SamplesPerPixel/PlanarConfiguration that make pydicom return 4D even
        though there's really only one channel; without this, downstream
        code raises "unsupported ndim=4".
        """
        if arr.ndim == 2:
            return arr[np.newaxis, ...]
        if arr.ndim == 3:
            return arr
        if arr.ndim == 4 and arr.shape[-1] == 1:
            return arr[..., 0]
        raise ValueError(f"Unsupported DICOM pixel array shape {arr.shape}")

    @staticmethod
    def _find_multiframes(meta_list):
        """
        Return every multi-frame candidate in the folder. Some non-image
        DICOMs (enhanced SR, etc.) also advertise NumberOfFrames > 1, so
        the caller must iterate and handle pixel_array failures — we can't
        tell from the header alone whether a candidate will decode.
        """
        return [
            (fp, ds) for fp, ds in meta_list
            if int(getattr(ds, "NumberOfFrames", 1)) > 1
        ]

    @staticmethod
    def _safe_pixel_array(ds, file_path: str = "") -> np.ndarray:
        """
        Read ds.pixel_array and wrap pydicom's opaque decoder failures in a
        clear, file-named ValueError. pydicom raises AttributeError from
        deep inside its decoder stack when required image tags are missing
        — catching that here means downstream sees a message that actually
        names the offending file.
        """
        try:
            return np.asarray(ds.pixel_array)
        except Exception as e:
            raise ValueError(
                f"Failed to decode pixel data from {file_path or '<unknown>'}: "
                f"{type(e).__name__}: {e}. The file is likely not an image "
                f"DICOM (e.g. Structured Report, Presentation State, or Dose "
                f"Report object)."
            ) from e

    @staticmethod
    def _is_image_dicom(ds) -> bool:
        """
        True if the dataset looks like a pixel-data-carrying image.
        Used to keep non-image DICOMs (Structured Reports, Presentation
        States, Dose Reports, etc.) out of the stack loop, since calling
        `.pixel_array` on them raises an opaque AttributeError from deep
        inside pydicom's decoder.

        Only Rows and Columns are checked — they're the two header fields
        that every image IOD in the DICOM standard requires. Including
        BitsAllocated was too strict and rejected some valid vendor DBT
        exports where that element wasn't surfaced by `stop_before_pixels`.
        """
        return hasattr(ds, "Rows") and hasattr(ds, "Columns")

    # -----------------------------
    # Public API (DO NOT break)
    # -----------------------------
    @staticmethod
    def load_series(folder_path: str):
        """
        Loads DICOM series from a folder.
        Returns: (volume, affine, original_shape, img_type)
        volume shape: (num_slices, rows, cols)

        Handles two layouts:
          - Multi-file series: one 2-D slice per .dcm file  (classic)
          - Single-file multi-frame: entire stack in one .dcm (e.g. DBT)
        """
        dicom_files = DicomLoader._list_dicom_files(folder_path)

        meta_list = [(fp, DicomLoader._read_meta(fp)) for fp in dicom_files]
        # Best-effort: drop non-image DICOMs (SR, presentation state, dose
        # report, etc.) that would crash on .pixel_array. But some vendor
        # exports don't surface Rows/Columns cleanly via stop_before_pixels,
        # so if the filter would empty the folder we keep every file and
        # let downstream read attempts speak for themselves — filtering is
        # an optimization, not a gate.
        filtered = [
            (fp, ds) for fp, ds in meta_list
            if DicomLoader._is_image_dicom(ds)
        ]
        if filtered:
            meta_list = filtered
        meta_list = DicomLoader._sort_series(meta_list)

        # ---- Multi-frame (e.g. DBT / tomosynthesis) ----
        # Try every multi-frame candidate. Header-level NumberOfFrames can
        # appear on non-image enhanced DICOMs too, so we can't trust the
        # first hit blindly. If a candidate fails to decode pixel data we
        # silently move on; if all of them fail we fall through to the
        # classic-stack path (also wrapped in try/except).
        for fp_mf, ds_mf_meta in DicomLoader._find_multiframes(meta_list):
            try:
                ds = pydicom.dcmread(fp_mf)
                # Keep native dtype (typically uint16) — downstream code
                # casts per slice, so a full-volume float32 copy would
                # waste ~2× RAM.
                volume = DicomLoader._normalize_pixel_array(
                    DicomLoader._safe_pixel_array(ds, fp_mf)
                )
                affine = DicomLoader._compute_affine_series(ds_mf_meta, None)
                original_shape = volume.shape
                return volume, affine, original_shape, DicomLoader.img_type
            except Exception:
                continue

        # ---- Classic multi-file series ----
        ds0 = meta_list[0][1]
        ds1 = meta_list[1][1] if len(meta_list) > 1 else None
        affine = DicomLoader._compute_affine_series(ds0, ds1)

        # Skip files that fail to decode as pixel data (non-image DICOMs
        # that slipped past the header filter). Keeps a mixed-content
        # folder usable instead of letting one bad file kill the series.
        slices = []
        for fp, _ in meta_list:
            try:
                ds = pydicom.dcmread(fp)
                slices.append(DicomLoader._safe_pixel_array(ds, fp))
            except Exception:
                continue

        if not slices:
            raise FileNotFoundError(
                f"No readable image slices in {folder_path}"
            )
        volume = np.stack(slices, axis=0)
        original_shape = volume.shape
        return volume, affine, original_shape, DicomLoader.img_type

    @staticmethod
    def load_slice(folder_path: str, slice_index: int | None = None):
        """
        Loads a single slice from a DICOM folder series.
        Returns: (slice_data, affine, original_shape, slice_index_used, img_type)

        Handles two layouts:
          - Multi-file series: one 2-D slice per .dcm file  (classic)
          - Single-file multi-frame: entire stack in one .dcm (e.g. DBT)
        """
        dicom_files = DicomLoader._list_dicom_files(folder_path)

        meta_list = [(fp, DicomLoader._read_meta(fp)) for fp in dicom_files]
        # Best-effort: drop non-image DICOMs (SR, presentation state, dose
        # report, etc.) that would crash on .pixel_array. But some vendor
        # exports don't surface Rows/Columns cleanly via stop_before_pixels,
        # so if the filter would empty the folder we keep every file and
        # let downstream read attempts speak for themselves — filtering is
        # an optimization, not a gate.
        filtered = [
            (fp, ds) for fp, ds in meta_list
            if DicomLoader._is_image_dicom(ds)
        ]
        if filtered:
            meta_list = filtered
        meta_list = DicomLoader._sort_series(meta_list)

        # ---- Multi-frame (e.g. DBT / tomosynthesis) ----
        # Same rule as load_series: iterate every multi-frame candidate,
        # return the first one that actually decodes. NumberOfFrames>1
        # can appear on non-image enhanced DICOMs, so we can't trust the
        # first hit without trying to read its pixels.
        for fp_mf, ds_mf_meta in DicomLoader._find_multiframes(meta_list):
            try:
                num_frames = int(ds_mf_meta.NumberOfFrames)
                if slice_index is None:
                    slice_index = num_frames // 2
                if slice_index < 0 or slice_index >= num_frames:
                    raise ValueError(
                        f"Slice index {slice_index} out of range. File has {num_frames} frames."
                    )
                ds = pydicom.dcmread(fp_mf)
                arr = DicomLoader._normalize_pixel_array(
                    DicomLoader._safe_pixel_array(ds, fp_mf)
                ).astype(np.float32)
                slice_data = arr[slice_index]
                affine = DicomLoader._compute_affine_series(ds_mf_meta, None)
                rows = int(getattr(ds, "Rows", slice_data.shape[0]))
                cols = int(getattr(ds, "Columns", slice_data.shape[1]))
                original_shape = (num_frames, rows, cols)
                return slice_data, affine, original_shape, slice_index, DicomLoader.img_type
            except Exception:
                # Reset slice_index for the next candidate — we may have
                # mutated it from None while validating the prior file.
                continue

        # ---- Classic multi-file series ----
        ds0 = meta_list[0][1]
        ds1 = meta_list[1][1] if len(meta_list) > 1 else None
        affine = DicomLoader._compute_affine_series(ds0, ds1)

        num_slices = len(meta_list)
        if slice_index is None:
            slice_index = num_slices // 2

        if slice_index < 0 or slice_index >= num_slices:
            raise ValueError(f"Slice index {slice_index} out of range. Series has {num_slices} slices.")

        # Try the requested index first, then walk the rest of the folder.
        # Some exports mix non-image DICOMs (SR, dose report, presentation
        # state) into the series — when the middle slot lands on one of
        # those, we fall back to any sibling that actually decodes. The
        # returned slice_index reflects which file we ended up using so
        # callers stay consistent with original_shape.
        tried = [slice_index] + [i for i in range(num_slices) if i != slice_index]
        last_error: Exception | None = None
        for idx in tried:
            target_fp = meta_list[idx][0]
            try:
                ds = pydicom.dcmread(target_fp)
                slice_data = DicomLoader._safe_pixel_array(ds, target_fp).astype(np.float32)
                rows = int(getattr(ds, "Rows", slice_data.shape[0]))
                cols = int(getattr(ds, "Columns", slice_data.shape[1]))
                original_shape = (num_slices, rows, cols)
                return slice_data, affine, original_shape, idx, DicomLoader.img_type
            except Exception as e:
                last_error = e
                continue

        raise FileNotFoundError(
            f"No decodable image slices in {folder_path} "
            f"(tried all {num_slices} files). Last error: {last_error}"
        )

    @staticmethod
    def load_single_file(file_path: str):
        """
        Loads a single DICOM file (2D slice or multi-frame).
        Returns: (volume, affine, original_shape, img_type)
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"DICOM file not found: {file_path}")

        if not file_path.lower().endswith((".dcm", ".dicom")):
            raise ValueError(f"Not a DICOM file extension: {file_path}")

        ds = pydicom.dcmread(file_path)
        if "PixelData" not in ds:
            raise ValueError(f"DICOM has no PixelData (not an image): {file_path}")

        # Normalize to 3D (k,r,c). Shared helper keeps this consistent with
        # the series loaders — previously each entry point had its own copy
        # and they drifted (the series path was missing the 4D squeeze,
        # which is what caused the "unsupported ndim=4" crash).
        volume = DicomLoader._normalize_pixel_array(
            DicomLoader._safe_pixel_array(ds, file_path)
        )
        original_shape = volume.shape

        # Best-effort affine for single-file (converted to RAS)
        affine = DicomLoader._compute_affine_series(ds0=ds, ds1=None)

        return volume, affine, original_shape, DicomLoader.img_type

    @staticmethod
    def load_single_slice(file_path: str, slice_index: int | None = None):
        """
        Loads a slice from a single DICOM file (2D or multi-frame).
        Returns: (slice_data, affine, original_shape, slice_index_used, img_type)
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"DICOM file not found: {file_path}")

        if not file_path.lower().endswith((".dcm", ".dicom")):
            raise ValueError(f"Not a DICOM file extension: {file_path}")

        ds = pydicom.dcmread(file_path)
        if "PixelData" not in ds:
            raise ValueError(f"DICOM has no PixelData: {file_path}")

        arr = DicomLoader._safe_pixel_array(ds, file_path)

        # Best-effort affine (converted to RAS)
        affine = DicomLoader._compute_affine_series(ds0=ds, ds1=None)

        # Case A: single 2D
        if arr.ndim == 2:
            if slice_index not in (None, 0):
                raise ValueError(f"slice_index={slice_index} invalid for 2D DICOM. Use 0.")
            slice_data = arr.astype(np.float32)
            original_shape = (1, slice_data.shape[0], slice_data.shape[1])
            return slice_data, affine, original_shape, 0, DicomLoader.img_type

        # Case B: multi-frame (k,r,c)
        if arr.ndim == 3:
            n_frames = arr.shape[0]
            if slice_index is None:
                slice_index = n_frames // 2
            if slice_index < 0 or slice_index >= n_frames:
                raise ValueError(f"Slice index {slice_index} out of range. File has {n_frames} frames.")
            slice_data = arr[slice_index].astype(np.float32)
            original_shape = (n_frames, slice_data.shape[0], slice_data.shape[1])
            return slice_data, affine, original_shape, slice_index, DicomLoader.img_type

        # Case C: 4D with single channel
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr3 = arr[..., 0]
            n_frames = arr3.shape[0]
            if slice_index is None:
                slice_index = n_frames // 2
            if slice_index < 0 or slice_index >= n_frames:
                raise ValueError(f"Slice index {slice_index} out of range. File has {n_frames} frames.")
            slice_data = arr3[slice_index].astype(np.float32)
            original_shape = (n_frames, slice_data.shape[0], slice_data.shape[1])
            return slice_data, affine, original_shape, slice_index, DicomLoader.img_type

        raise ValueError(f"Unsupported DICOM pixel array ndim={arr.ndim}")


class UnifiedImageLoader:
    """
    Unified interface for loading medical images in their NATIVE orientation.
    """

    @staticmethod
    def load_image(file_path: str):
        """
        Load complete volume in native orientation.
        Returns: (volume, affine, original_shape, img_type)
        """
        if os.path.isdir(file_path):
            return DicomLoader.load_series(file_path)
        if file_path.lower().endswith((".dcm", ".dicom")):
            return DicomLoader.load_single_file(file_path)
        if file_path.lower().endswith((".nii", ".nii.gz")):
            return NiftiLoader.load_volume(file_path)
        raise ValueError(f"Unsupported format: {file_path}")

    @staticmethod
    def load_slice(file_path: str, slice_index: int | None = None):
        """
        Load a single slice in native orientation.
        Returns: (slice_data, affine, original_shape, slice_index_used, img_type)
        """
        if os.path.isdir(file_path):
            return DicomLoader.load_slice(file_path, slice_index)
        if file_path.lower().endswith((".dcm", ".dicom")):
            return DicomLoader.load_single_slice(file_path, slice_index)
        if file_path.lower().endswith((".nii", ".nii.gz")):
            return NiftiLoader.load_slice(file_path, slice_index)
        raise ValueError(f"Unsupported format: {file_path}")

    @staticmethod
    def get_slice_dimension(original_shape):
        """
        Identifies which dimension contains slices (usually the smallest).
        """
        if len(original_shape) < 3:
            return 0
        return int(np.argmin(original_shape))
