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
        
        Args:
            shape: tuple of 3D volume dimensions
            
        Returns:
            int: central slice index along the slice dimension
        """
        slice_dim = np.argmin(shape)
        return shape[slice_dim] // 2

class NiftiLoader(BaseImageLoader):
    """Specialized loader for NIfTI files - loads in NATIVE orientation."""
    
    @staticmethod
    def load_volume(file_path, lazy=False):
        """
        Loads NIfTI volume in its NATIVE orientation.
        
        Args:
            file_path: path to NIfTI file
            lazy: if True, uses lazy loading
            
        Returns:
            tuple: (data, affine, original_shape)
                - data: numpy array in NATIVE orientation
                - affine: 4x4 affine transformation matrix
                - original_shape: shape of the volume (same as data.shape)
        """
        img = nib.load(file_path)
        if lazy:
            data = np.asarray(img.dataobj)  # Lazy loading
        else:
            data = img.get_fdata()
        
        # NO rearrangement - keep native orientation
        original_shape = data.shape
        return data, img.affine, original_shape
    
    @staticmethod
    def load_slice(file_path, slice_index=None):
        """
        Loads a specific slice from NIfTI file in NATIVE orientation.
        
        Args:
            file_path (str): Path to NIfTI file
            slice_index (int, optional): Specific slice index. If None, loads central slice.
            
        Returns:
            tuple: (slice_data, affine, original_shape, slice_index_used)
                - slice_data: 2D numpy array of the slice in NATIVE orientation
                - affine: 4x4 affine matrix
                - original_shape: 3D shape of the full volume
                - slice_index_used: which slice was actually loaded
        """
        img = nib.load(file_path)
        original_shape = img.shape
        
        # Handle 2D images
        if len(original_shape) == 2:
            slice_data = img.get_fdata()
            return slice_data, img.affine, original_shape, 0
        
        # For 3D volumes, find slice dimension (smallest dimension)
        slice_dim = np.argmin(original_shape)
        
        # Calculate central slice if not specified
        if slice_index is None:
            slice_index = original_shape[slice_dim] // 2
        
        # Validate slice index
        if slice_index < 0 or slice_index >= original_shape[slice_dim]:
            raise ValueError(f"Slice index {slice_index} out of range for dimension {slice_dim} with size {original_shape[slice_dim]}")
        
        # Load only the required slice using memory mapping (efficient)
        # Extract slice based on which dimension contains slices
        if slice_dim == 0:
            slice_data = np.asarray(img.dataobj[slice_index, :, :])
        elif slice_dim == 1:
            slice_data = np.asarray(img.dataobj[:, slice_index, :])
        else:  # slice_dim == 2
            slice_data = np.asarray(img.dataobj[:, :, slice_index])
        
        return slice_data, img.affine, original_shape, slice_index

class DicomLoader(BaseImageLoader):
    """Specialized loader for DICOM series - loads in NATIVE orientation."""
    
    @staticmethod
    def load_series(folder_path):
        """
        Loads DICOM series in NATIVE orientation (no rearrangement).
        
        Args:
            folder_path: path to folder containing DICOM files
            
        Returns:
            tuple: (volume, affine, original_shape)
        """
        # Load DICOM files
        dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                      if f.lower().endswith(('.dcm', '.dicom'))]
        
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {folder_path}")
        
        # Sort and load slices
        dicoms = [pydicom.dcmread(f) for f in dicom_files]
        dicoms.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))
        
        # Create 3D volume from slices - native stacking (no rearrangement)
        volume = np.stack([d.pixel_array for d in dicoms], axis=0)
        
        # Return identity matrix as affine for DICOM (could compute real affine from metadata)
        original_shape = volume.shape
        return volume, np.eye(4), original_shape
    
    @staticmethod
    def load_slice(folder_path, slice_index=None):
        """
        Loads a specific slice from DICOM series.
        
        Args:
            folder_path (str): Path to DICOM directory
            slice_index (int, optional): Specific slice index. If None, loads central slice.
            
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
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            dicoms.append((file_path, ds))
        
        dicoms.sort(key=lambda x: int(getattr(x[1], "InstanceNumber", 0)))
        
        # Get total number of slices and image dimensions
        num_slices = len(dicoms)
        
        # Calculate central slice if not specified
        if slice_index is None:
            slice_index = num_slices // 2
        elif slice_index >= num_slices:
            raise ValueError(f"Slice index {slice_index} out of range. Series has {num_slices} slices.")
        
        # Load only the specific slice
        target_file = dicoms[slice_index][0]
        ds = pydicom.dcmread(target_file)
        slice_data = ds.pixel_array.astype(np.float32)
        
        # Get dimensions from first slice to build original_shape
        rows = ds.Rows
        cols = ds.Columns
        original_shape = (num_slices, rows, cols)
        
        return slice_data, np.eye(4), original_shape, slice_index

    @staticmethod
    def load_single_file(file_path):
        """
        Loads a single DICOM file (2D slice or multi-frame volume).

        Args:
            file_path (str): Path to DICOM file (.dcm, .dicom)

        Returns:
            tuple: (volume, affine, original_shape)
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"DICOM file not found: {file_path}")

        if not file_path.lower().endswith((".dcm", ".dicom")):
            raise ValueError(f"Not a DICOM file extension: {file_path}")

        ds = pydicom.dcmread(file_path)

        if "PixelData" not in ds:
            raise ValueError(f"DICOM has no PixelData (not an image): {file_path}")

        arr = ds.pixel_array

        # Normalize to 3D volume
        if arr.ndim == 2:
            volume = arr[np.newaxis, ...].astype(np.float32)  # (1, H, W)
        elif arr.ndim == 3:
            volume = arr.astype(np.float32)  # (n_frames, H, W)
        elif arr.ndim == 4 and arr.shape[-1] == 1:
            volume = arr[..., 0].astype(np.float32)
        else:
            raise ValueError(f"Unsupported DICOM pixel array shape {arr.shape}")

        original_shape = volume.shape
        affine = np.eye(4, dtype=np.float32)

        return volume, affine, original_shape

    @staticmethod
    def load_single_slice(file_path, slice_index=None):
        """
        Loads a slice from a single DICOM file.
        
        Args:
            file_path: path to DICOM file
            slice_index: specific slice/frame index
            
        Returns:
            tuple: (slice_data, affine, original_shape, slice_index_used)
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"DICOM file not found: {file_path}")

        if not file_path.lower().endswith((".dcm", ".dicom")):
            raise ValueError(f"Not a DICOM file extension: {file_path}")

        ds = pydicom.dcmread(file_path)

        if "PixelData" not in ds:
            raise ValueError(f"DICOM has no PixelData: {file_path}")

        arr = ds.pixel_array

        # Case A: Single 2D image
        if arr.ndim == 2:
            if slice_index is None or slice_index == 0:
                slice_index_used = 0
            else:
                raise ValueError(f"slice_index={slice_index} invalid for 2D DICOM. Use 0.")
            
            slice_data = arr.astype(np.float32)
            original_shape = (1, slice_data.shape[0], slice_data.shape[1])
            return slice_data, np.eye(4, dtype=np.float32), original_shape, slice_index_used

        # Case B: Multi-frame 3D
        if arr.ndim == 3:
            n_frames = arr.shape[0]
            
            if slice_index is None:
                slice_index_used = n_frames // 2
            else:
                if slice_index < 0 or slice_index >= n_frames:
                    raise ValueError(f"Slice index {slice_index} out of range. File has {n_frames} frames.")
                slice_index_used = slice_index
            
            slice_data = arr[slice_index_used].astype(np.float32)
            original_shape = (n_frames, slice_data.shape[0], slice_data.shape[1])
            return slice_data, np.eye(4, dtype=np.float32), original_shape, slice_index_used

        # Case C: 4D with single channel
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
            n_frames = arr.shape[0]
            slice_index_used = n_frames // 2 if slice_index is None else slice_index
            
            if slice_index_used < 0 or slice_index_used >= n_frames:
                raise ValueError(f"Slice index {slice_index_used} out of range. File has {n_frames} frames.")
            
            slice_data = arr[slice_index_used].astype(np.float32)
            original_shape = (n_frames, slice_data.shape[0], slice_data.shape[1])
            return slice_data, np.eye(4, dtype=np.float32), original_shape, slice_index_used

        raise ValueError(f"Unsupported DICOM pixel array ndim={arr.ndim}")


class UnifiedImageLoader:
    """
    Unified interface for loading medical images in their NATIVE orientation.
    
    Key principle: Load images as-is, apply transformations only for display,
    and always save in native orientation with correct affine matrix.
    """

    @staticmethod
    def load_image(file_path):
        """
        Load complete volume in native orientation.
        
        Returns:
            tuple: (volume, affine, original_shape)
        """
        if os.path.isdir(file_path):
            return DicomLoader.load_series(file_path)
        elif file_path.lower().endswith(('.dcm', '.dicom')):
            return DicomLoader.load_single_file(file_path)
        elif file_path.lower().endswith(('.nii', '.nii.gz')):
            return NiftiLoader.load_volume(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_path}")
        
    @staticmethod
    def load_slice(file_path, slice_index=None):
        """
        Load a specific slice in native orientation.
        
        Args:
            file_path: path to image file or DICOM directory
            slice_index: specific slice index (None = central slice)
            
        Returns:
            tuple: (slice_data, affine, original_shape, slice_index_used)
        """
        if os.path.isdir(file_path):
            return DicomLoader.load_slice(file_path, slice_index)
        elif file_path.lower().endswith(('.dcm', '.dicom')):
            return DicomLoader.load_single_slice(file_path, slice_index)
        elif file_path.lower().endswith(('.nii', '.nii.gz')):
            return NiftiLoader.load_slice(file_path, slice_index)
        else:
            raise ValueError(f"Unsupported format: {file_path}")
    
    @staticmethod
    def get_slice_dimension(original_shape):
        """
        Identifies which dimension contains slices (usually the smallest).
        
        Args:
            original_shape: 3D shape tuple
            
        Returns:
            int: index of slice dimension (0, 1, or 2)
        """
        return np.argmin(original_shape)