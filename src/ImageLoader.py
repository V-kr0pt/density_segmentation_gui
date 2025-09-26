import os
import numpy as np
import pydicom
import nibabel as nib

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
        shape = img.shape
        if len(shape) == 2:
            # Already 2D, just load and return
            slice_data = img.get_fdata()
            slice_data, original_shape = BaseImageLoader.rearrange_dimensions(slice_data[np.newaxis, ...])
            slice_data = slice_data[0]  # Remove singleton dimension
            return slice_data, img.affine, original_shape, 0
        
        # For 3D/4D volumes, determine slice dimension and index
        rearranged_shape, slice_dim = NiftiLoader._predict_slice_dimension(shape)
        
        # Calculate central slice if not specified
        if slice_index is None:
            slice_index = rearranged_shape[slice_dim] // 2
        
        # Load only the required slice using memory mapping
        if len(shape) == 3:
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
        rearranged, original_shape = BaseImageLoader.rearrange_dimensions(slice_2d)
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

class UnifiedImageLoader:
    """Unified Interface for all formats"""

    @staticmethod
    def load_image(file_path):
        if os.path.isdir(file_path):
            return DicomLoader.load_series(file_path)
        elif file_path.lower().endswith(('.nii', '.nii.gz')):
            return NiftiLoader.load_volume(file_path)
        else:
            raise ValueError(f"Formato não suportado: {file_path}")
        
    @staticmethod
    def load_slice(file_path, slice_index=None):
        """
        Optimized method to load only a specific slice without loading entire volume.
        
        Args:
            file_path (str): Path to NIfTI file or DICOM directory
            slice_index (int, optional): Specific slice index. If None, loads central slice.
            orientation_rules (dict, optional): Rules for image orientation
            
        Returns:
            tuple: (slice_data, affine, original_shape, slice_index_used)
        """
        if os.path.isdir(file_path):
            return DicomLoader.load_slice(file_path, slice_index)
        elif file_path.lower().endswith(('.nii', '.nii.gz')):
            return NiftiLoader.load_slice(file_path, slice_index)
        else:
            raise ValueError(f"Formato não suportado: {file_path}")