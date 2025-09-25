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
        slice_dim = np.argmin(data.shape)

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

        return rearranged_data
    
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
        data = BaseImageLoader.rearrange_dimensions(data)
        return data, img.affine

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
        volume = BaseImageLoader.rearrange_dimensions(volume)
        return volume, np.eye(4)  # Return identity matrix as affine for DICOM

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
        
    def load_slice(slice_index=None, orientation_rules=None):
        ...