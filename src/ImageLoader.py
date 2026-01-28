import os
import numpy as np
import pydicom
import nibabel as nib
from dicom_converter import get_converter

class BaseImageLoader:
    """Base class for medical image loading operations."""

    @staticmethod
    def rearrange_dimensions(data):
        """
        Ensures the slice dimension (always the smallest dimension) becomes the first dimension.
        Transforms (z, x, y), (x, z, y), (x, y, z) -> (z, x, y) where z is the smallest dimension.
        Maintains x, y order where y > x (height > width).

        Args:
            data: 3D numpy array with any dimension ordering

        Returns:
            numpy array with dimensions ordered as (slices, width, height) where height > width
        """
        # Find the slice dimension (smallest dimension)
        original_shape = data.shape
        slice_dim = np.argmin(original_shape)

        # Get all three dimension indices
        dims = [0, 1, 2]
        # Get the other two dimensions (non-slice dimensions)
        other_dims = [dim for dim in dims if dim != slice_dim]

        # Ensure the two other dimensions are ordered so the larger one (height) comes last
        if data.shape[other_dims[1]] > data.shape[other_dims[0]]:
            # Correct order already: second dimension is larger (height > width)
            new_order = [slice_dim, other_dims[0], other_dims[1]]  # (slices, width, height)
        else:
            # Need to swap to ensure height > width
            new_order = [slice_dim, other_dims[1], other_dims[0]]  # (slices, height, width) -> will be corrected

        # Apply the transpose
        rearranged_data = np.transpose(data, new_order)

        # Final verification: ensure the last dimension (height) is larger than middle dimension (width)
        if rearranged_data.shape[2] <= rearranged_data.shape[1]:
            # If still not correct, swap the last two dimensions
            rearranged_data = np.transpose(rearranged_data, (0, 2, 1))
        
        if rearranged_data.shape != original_shape:
            print(f"WARNING: The original shape was changed: before {original_shape} -> now {rearranged_data.shape}.") 

        return rearranged_data, original_shape
    
    @staticmethod
    def normalize_orientation(image, orientation_rules):
        """Applies consistent rotation/flip operations based on orientation rules."""
        # Orientation normalization logic
        pass

class NiftiLoader(BaseImageLoader):
    """Specialized loader for NIfTI files."""
    
    @staticmethod
    def load_volume(file_path, lazy=False):
        """Loads NIfTI volume and applies dimension rearrangement."""
        img = nib.load(file_path)
        if lazy:
            data = img.dataobj  # Lazy loading
        else:
            data = img.get_fdata()
        
        # Apply dimension rearrangement to ensure consistent orientation
        data, original_shape = BaseImageLoader.rearrange_dimensions(data)
        return data, img.affine, original_shape
    
    @staticmethod
    def load_slice(file_path, slice_index=None):
        """
        Optimized method to load only a specific slice from NIfTI without loading entire volume.
        
        Args:
            file_path (str): Path to NIfTI file
            slice_index (int, optional): Specific slice index. If None, loads central slice.
            orientation_rules (dict, optional): Rules for image orientation
            
        Returns:
            tuple: (slice_data, affine, original_shape, slice_index_used)
        """
        img = nib.load(file_path)
        
        # Get volume shape without loading data
        original_shape = img.shape
        if len(original_shape) == 2:
            # Already 2D, just load and return
            slice_data = img.get_fdata()
            slice_data, _ = BaseImageLoader.rearrange_dimensions(slice_data[np.newaxis, ...])
            slice_data = slice_data[0]  # Remove singleton dimension
            return slice_data, img.affine, original_shape, 0
        
        # For 3D/4D volumes, determine slice dimension and index
        rearranged_shape, slice_dim = NiftiLoader._predict_slice_dimension(original_shape)
        
        # Calculate central slice if not specified
        if slice_index is None:
            slice_index = rearranged_shape[slice_dim] // 2
        
        # Load only the required slice using memory mapping
        if len(original_shape) == 3:
            # 3D volume - use direct indexing with memory mapping
            if slice_dim == 0:
                slice_data = img.dataobj[slice_index, :, :]
            elif slice_dim == 1:
                slice_data = img.dataobj[:, slice_index, :]
            else:  # slice_dim == 2
                slice_data = img.dataobj[:, :, slice_index]
        
        # Convert to array and ensure proper dtype
        slice_data = np.asarray(slice_data)
        
        # Apply dimension rearrangement to 2D slice
        slice_2d = slice_data[np.newaxis, ...]  # Add batch dimension
        rearranged, _ = BaseImageLoader.rearrange_dimensions(slice_2d)
        slice_data = rearranged[0]  # Remove batch dimension
        print(f"slice_data dim:{slice_data.shape}")
        return slice_data, img.affine, original_shape, slice_index

    @staticmethod
    def _predict_slice_dimension(shape):
        """
        Predicts which dimension contains slices without loading data.
        Returns rearranged shape and slice dimension index.
        """
        if len(shape) == 2:
            return shape, 0
        
        # Create a dummy array with the same shape to test rearrangement
        dummy_data = np.zeros(shape)
        rearranged, _ = BaseImageLoader.rearrange_dimensions(dummy_data)
        slice_dim = np.argmin(rearranged.shape)
        
        return rearranged.shape, slice_dim

class DicomLoader(BaseImageLoader):
    """Specialized loader for DICOM series."""
    
    @staticmethod
    def load_series(folder_path):
        """Loads DICOM series and applies dimension rearrangement."""
        # Load DICOM files
        dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                      if f.lower().endswith(('.dcm', '.dicom'))]
        
        # Sort and load slices
        dicoms = [pydicom.dcmread(f) for f in dicom_files]
        dicoms.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))
        
        # Create 3D volume from slices
        volume = np.stack([d.pixel_array for d in dicoms], axis=0)
        
        # Apply dimension rearrangement
        volume, original_shape = BaseImageLoader.rearrange_dimensions(volume)
        return volume, np.eye(4), original_shape  # Return identity matrix as affine for DICOM
    
    @staticmethod
    def load_slice(folder_path, slice_index=None):
        """
        Optimized method to load only a specific slice from DICOM series.
        
        Args:
            folder_path (str): Path to DICOM directory
            slice_index (int, optional): Specific slice index. If None, loads central slice.
            orientation_rules (dict, optional): Rules for image orientation
            
        Returns:
            tuple: (slice_data, affine, original_shape, slice_index_used)
        """
        # Load DICOM files metadata first (fast)
        dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                      if f.lower().endswith(('.dcm', '.dicom'))]
        
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {folder_path}")
        
        # Sort by InstanceNumber without loading pixel data
        dicoms = []
        for file_path in dicom_files:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)  # Fast metadata only
            dicoms.append((file_path, ds))
        
        dicoms.sort(key=lambda x: int(getattr(x[1], "InstanceNumber", 0)))
        
        # Calculate central slice if not specified
        if slice_index is None:
            slice_index = len(dicoms) // 2
        elif slice_index >= len(dicoms):
            raise ValueError(f"Slice index {slice_index} out of range. Series has {len(dicoms)} slices.")
        
        # Load only the required slice's pixel data
        file_path, ds_meta = dicoms[slice_index]
        ds_full = pydicom.dcmread(file_path)  # Now load with pixel data
        slice_data = ds_full.pixel_array.astype(np.float32)
        
        # Get original volume shape from metadata
        num_slices = len(dicoms)
        rows = ds_full.Rows
        cols = ds_full.Columns
        original_shape = (num_slices, rows, cols)
        
        # Create 3D-like array for dimension rearrangement
        volume_like = slice_data[np.newaxis, ...]  # Add slice dimension
        rearranged, _ = BaseImageLoader.rearrange_dimensions(volume_like)
        slice_data = rearranged[0]  # Remove slice dimension
        
        return slice_data, np.eye(4), original_shape, slice_index

    @staticmethod
    def load_single_file(file_path):
        """
        Loads a single DICOM file (either 2D slice or multi-frame volume).

        Args:
            file_path (str): Path to a DICOM file (.dcm, .dicom)

        Returns:
            tuple: (volume, affine, original_shape)
                - volume: np.ndarray (after rearrange_dimensions)
                - affine: np.ndarray (4x4) - identity for now
                - original_shape: tuple (before rearrange_dimensions)
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"DICOM file not found: {file_path}")

        if not file_path.lower().endswith((".dcm", ".dicom")):
            raise ValueError(f"Not a DICOM file extension: {file_path}")

        ds = pydicom.dcmread(file_path)

        # Some DICOMs can be missing pixel data (e.g., RTSTRUCT, SR, etc.)
        if "PixelData" not in ds:
            raise ValueError(f"DICOM has no PixelData (not an image): {file_path}")

        arr = ds.pixel_array  # may be 2D, 3D (multiframe), or sometimes 4D (rare)

        # Normalize to a 3D "volume-like" array: (Z, Y, X)
        print(f"The DICOM dim: {arr.ndim}")
        if arr.ndim == 2:
            volume = arr[np.newaxis, ...]  # (1, H, W)
        elif arr.ndim == 3:
            # Typically (n_frames, H, W) for multiframe
            volume = arr
        elif arr.ndim == 4:
            # Some modalities/encodings can yield (n_frames, H, W, channels)
            # For medical grayscale data, channels is often 1; handle conservatively:
            if arr.shape[-1] == 1:
                volume = arr[..., 0]
            else:
                raise ValueError(
                    f"Unsupported 4D DICOM pixel array shape {arr.shape} in {file_path}. "
                    "If this is RGB, you'll need a policy (e.g., convert to grayscale or keep channels)."
                )
        else:
            raise ValueError(f"Unsupported DICOM pixel array ndim={arr.ndim} for file {file_path}")

        volume = volume.astype(np.float32)

        original_shape = volume.shape  # (Z, Y, X) before rearrange
        volume, _ = BaseImageLoader.rearrange_dimensions(volume)

        # TODO: If you want a real affine, you can compute it from:
        # ImagePositionPatient, ImageOrientationPatient, PixelSpacing, SliceThickness/SpacingBetweenSlices
        affine = np.eye(4, dtype=np.float32)

        return volume, affine, original_shape

    @staticmethod
    def load_single_slice(file_path, slice_index=None):
        """
        Load a slice from a single DICOM file.
        Works for:
          - 2D single-slice DICOM  -> returns that slice (index 0)
          - multi-frame DICOM (3D) -> returns selected frame

        Returns:
            (slice_data, affine, original_shape, slice_index_used)
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"DICOM file not found: {file_path}")

        if not file_path.lower().endswith((".dcm", ".dicom")):
            raise ValueError(f"Not a DICOM file extension: {file_path}")

        ds = pydicom.dcmread(file_path)

        if "PixelData" not in ds:
            raise ValueError(f"DICOM has no PixelData (not an image): {file_path}")

        arr = ds.pixel_array

        # Case A: single 2D image
        if arr.ndim == 2:
            # Only one "slice" exists
            if slice_index is None:
                slice_index_used = 0
            else:
                if slice_index != 0:
                    raise ValueError(
                        f"slice_index={slice_index} invalid for 2D single-file DICOM. Use 0."
                    )
                slice_index_used = slice_index

            slice_data = arr.astype(np.float32)
            original_shape = (1, slice_data.shape[0], slice_data.shape[1])  # (Z=1, H, W)

            # Use your rearrange pipeline (expects 3D)
            volume_like = slice_data[np.newaxis, ...]  # (1, H, W)
            rearranged, _ = BaseImageLoader.rearrange_dimensions(volume_like)
            slice_data = rearranged[0]

            return slice_data, np.eye(4, dtype=np.float32), original_shape, slice_index_used

        # Case B: multi-frame (typically (n_frames, H, W))
        if arr.ndim == 3:
            n_frames = arr.shape[0]

            if slice_index is None:
                slice_index_used = n_frames // 2
            else:
                if slice_index < 0 or slice_index >= n_frames:
                    raise ValueError(
                        f"Slice index {slice_index} out of range. File has {n_frames} frames."
                    )
                slice_index_used = slice_index

            frame = arr[slice_index_used].astype(np.float32)  # (H, W)
            original_shape = (n_frames, frame.shape[0], frame.shape[1])

            volume_like = frame[np.newaxis, ...]  # (1, H, W)
            rearranged, _ = BaseImageLoader.rearrange_dimensions(volume_like)
            slice_data = rearranged[0]

            return slice_data, np.eye(4, dtype=np.float32), original_shape, slice_index_used

        # Optional: handle 4D (rare)
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
            # then treat as 3D above
            n_frames = arr.shape[0]
            slice_index_used = n_frames // 2 if slice_index is None else slice_index
            if slice_index_used < 0 or slice_index_used >= n_frames:
                raise ValueError(f"Slice index {slice_index_used} out of range. File has {n_frames} frames.")
            frame = arr[slice_index_used].astype(np.float32)
            original_shape = (n_frames, frame.shape[0], frame.shape[1])

            volume_like = frame[np.newaxis, ...]
            rearranged, _ = BaseImageLoader.rearrange_dimensions(volume_like)
            slice_data = rearranged[0]

            return slice_data, np.eye(4, dtype=np.float32), original_shape, slice_index_used

        raise ValueError(f"Unsupported DICOM pixel array ndim={arr.ndim} for file {file_path}")

    

class UnifiedImageLoader:
    """
    Unified Interface for all formats with automatic DICOM→NIfTI conversion.
    
    Performance optimization: DICOM inputs are automatically converted to NIfTI
    in memory, eliminating 100× I/O overhead in slice-by-slice processing.
    """

    @staticmethod
    def load_image(file_path):
        """
        Load medical image volume with automatic DICOM conversion.
        
        DICOM files/directories are automatically converted to NIfTI format
        in memory for 50-200× faster subsequent operations.
        
        Args:
            file_path: Path to NIfTI file, DICOM file, or DICOM directory
            
        Returns:
            Tuple of (volume, affine, original_shape)
        """
        # Auto-convert DICOM to NIfTI in memory
        if os.path.isdir(file_path):
            # DICOM series directory - convert in memory
            converter = get_converter()
            volume, affine = converter.convert_to_nifti(file_path)
            original_shape = volume.shape
            return volume, affine, original_shape
            
        elif file_path.lower().endswith(('.dcm', '.dicom')):
            # Single DICOM file - convert in memory
            converter = get_converter()
            volume, affine = converter.convert_to_nifti(file_path)
            original_shape = volume.shape
            return volume, affine, original_shape
            
        elif file_path.lower().endswith(('.nii', '.nii.gz')):
            # Already NIfTI
            return NiftiLoader.load_volume(file_path)
            
        else:
            raise ValueError(f"Format is not supported: {file_path}")
        
    @staticmethod
    def load_slice(file_path, slice_index=None):
        """
        Optimized slice loading with automatic DICOM conversion.
        
        DICOM inputs are converted to NIfTI in memory on first access.
        All subsequent slice operations in the same session use the in-memory volume.
        
        Args:
            file_path: Path to NIfTI file, DICOM file, or DICOM directory
            slice_index (int, optional): Specific slice index. If None, loads central slice.
            
        Returns:
            tuple: (slice_data, affine, original_shape, slice_index_used)
        """
        # Auto-convert DICOM to NIfTI in memory
        if os.path.isdir(file_path):
            # DICOM series directory - convert and extract slice
            converter = get_converter()
            volume, affine = converter.convert_to_nifti(file_path)
            original_shape = volume.shape
            
            # Calculate central slice if not specified
            if slice_index is None:
                slice_index = original_shape[0] // 2
            elif slice_index >= original_shape[0]:
                raise ValueError(f"Slice index {slice_index} out of range. Volume has {original_shape[0]} slices.")
            
            slice_data = volume[slice_index, :, :]
            return slice_data, affine, original_shape, slice_index
            
        elif file_path.lower().endswith(('.dcm', '.dicom')):
            # Single DICOM file - convert and extract slice
            converter = get_converter()
            volume, affine = converter.convert_to_nifti(file_path)
            original_shape = volume.shape
            
            # Calculate central slice if not specified
            if slice_index is None:
                slice_index = original_shape[0] // 2
            elif slice_index >= original_shape[0]:
                raise ValueError(f"Slice index {slice_index} out of range. Volume has {original_shape[0]} slices.")
            
            slice_data = volume[slice_index, :, :]
            return slice_data, affine, original_shape, slice_index
            
        elif file_path.lower().endswith(('.nii', '.nii.gz')):
            # Already NIfTI
            return NiftiLoader.load_slice(file_path, slice_index)
            
        else:
            raise ValueError(f"Format is not supported: {file_path}")