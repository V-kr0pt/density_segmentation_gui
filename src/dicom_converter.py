"""
DICOM to NIfTI Conversion Module
Simple in-memory conversion without caching complexity.
"""

import os
import numpy as np
import nibabel as nib
import pydicom
from typing import Tuple, Dict, Any


class DicomToNiftiConverter:
    """
    Handles DICOM to NIfTI conversion with in-memory processing.
    
    Design philosophy: Simple and robust over complex caching.
    Converts DICOM → NIfTI in memory, writes to temp output location.
    """
    
    def _compute_affine_from_dicom(self, dicom_datasets: list) -> np.ndarray:
        """
        Compute accurate affine transformation matrix from DICOM metadata.
        
        Uses ImagePositionPatient, ImageOrientationPatient, and PixelSpacing
        to create proper spatial transformation matrix.
        
        Args:
            dicom_datasets: List of sorted DICOM datasets
            
        Returns:
            4x4 affine matrix (numpy array)
        """
        try:
            ds = dicom_datasets[0]  # Reference slice
            
            # Get orientation (direction cosines)
            orientation = np.array(ds.ImageOrientationPatient).reshape(2, 3)
            row_cosine = orientation[0]  # X direction
            col_cosine = orientation[1]  # Y direction
            
            # Compute slice direction (cross product)
            slice_cosine = np.cross(row_cosine, col_cosine)
            
            # Get pixel spacing
            pixel_spacing = np.array(ds.PixelSpacing)  # [row_spacing, col_spacing]
            
            # Compute slice spacing
            if len(dicom_datasets) > 1:
                pos1 = np.array(dicom_datasets[0].ImagePositionPatient)
                pos2 = np.array(dicom_datasets[1].ImagePositionPatient)
                slice_spacing = np.linalg.norm(pos2 - pos1)
            else:
                # Single slice: try SliceThickness or default to 1.0
                slice_spacing = float(getattr(ds, 'SliceThickness', 1.0))
            
            # Get origin (position of first voxel)
            origin = np.array(ds.ImagePositionPatient)
            
            # Build affine matrix
            # Columns: [X_direction, Y_direction, Z_direction, origin]
            affine = np.eye(4)
            affine[:3, 0] = row_cosine * pixel_spacing[1]  # X direction scaled
            affine[:3, 1] = col_cosine * pixel_spacing[0]  # Y direction scaled
            affine[:3, 2] = slice_cosine * slice_spacing   # Z direction scaled
            affine[:3, 3] = origin                          # Origin position
            
            return affine
            
        except (AttributeError, KeyError, IndexError) as e:
            # Fallback to identity matrix if metadata is incomplete
            print(f"Warning: Could not compute affine from DICOM metadata ({e}). Using identity.")
            return np.eye(4)
    
    def _convert_dicom_series(self, folder_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Convert DICOM series (directory) to NIfTI volume.
        
        Args:
            folder_path: Path to directory containing DICOM files
            
        Returns:
            Tuple of (volume_array, affine_matrix, metadata_dict)
        """
        # Load DICOM files
        dicom_files = [
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.dcm', '.dicom'))
        ]
        
        if not dicom_files:
            raise FileNotFoundError(f"No DICOM files found in {folder_path}")
        
        # Read and sort slices
        dicoms = [pydicom.dcmread(f) for f in dicom_files]
        dicoms.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))
        
        # Stack into 3D volume
        volume = np.stack([d.pixel_array.astype(np.float32) for d in dicoms], axis=0)
        
        # Compute affine matrix
        affine = self._compute_affine_from_dicom(dicoms)
        
        # Extract metadata
        ds = dicoms[0]
        metadata = {
            'source_type': 'dicom_series',
            'num_slices': len(dicoms),
            'patient_id': str(getattr(ds, 'PatientID', 'Unknown')),
            'study_description': str(getattr(ds, 'StudyDescription', 'Unknown')),
            'series_description': str(getattr(ds, 'SeriesDescription', 'Unknown')),
            'modality': str(getattr(ds, 'Modality', 'Unknown')),
            'pixel_spacing': list(getattr(ds, 'PixelSpacing', [1.0, 1.0])),
            'slice_thickness': float(getattr(ds, 'SliceThickness', 1.0)),
            'original_shape': volume.shape
        }
        
        return volume, affine, metadata
    
    def _convert_dicom_single_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Convert single DICOM file to NIfTI volume.
        
        Args:
            file_path: Path to DICOM file
            
        Returns:
            Tuple of (volume_array, affine_matrix, metadata_dict)
        """
        ds = pydicom.dcmread(file_path)
        
        if "PixelData" not in ds:
            raise ValueError(f"DICOM has no PixelData: {file_path}")
        
        arr = ds.pixel_array.astype(np.float32)
        
        # Normalize to 3D
        if arr.ndim == 2:
            volume = arr[np.newaxis, ...]  # (1, H, W)
        elif arr.ndim == 3:
            volume = arr  # Multi-frame (N, H, W)
        else:
            raise ValueError(f"Unsupported DICOM array shape: {arr.shape}")
        
        # Compute affine (single file, simplified)
        try:
            affine = self._compute_affine_from_dicom([ds])
        except:
            affine = np.eye(4)
        
        # Extract metadata
        metadata = {
            'source_type': 'dicom_single_file',
            'num_slices': volume.shape[0],
            'patient_id': str(getattr(ds, 'PatientID', 'Unknown')),
            'study_description': str(getattr(ds, 'StudyDescription', 'Unknown')),
            'series_description': str(getattr(ds, 'SeriesDescription', 'Unknown')),
            'modality': str(getattr(ds, 'Modality', 'Unknown')),
            'pixel_spacing': list(getattr(ds, 'PixelSpacing', [1.0, 1.0])),
            'original_shape': volume.shape
        }
        
        return volume, affine, metadata
    
    def convert_to_nifti(self, dicom_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert DICOM to NIfTI format in memory (no caching).
        
        Simple and robust: converts once per session, no disk cache complexity.
        
        Args:
            dicom_path: Path to DICOM file or directory
            
        Returns:
            Tuple of (volume_array, affine_matrix)
            
        Raises:
            FileNotFoundError: If DICOM path doesn't exist
            ValueError: If DICOM format is invalid
        """
        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"DICOM path not found: {dicom_path}")
        
        # Convert DICOM based on type
        if os.path.isdir(dicom_path):
            volume, affine, _ = self._convert_dicom_series(dicom_path)
        else:
            volume, affine, _ = self._convert_dicom_single_file(dicom_path)
        
        return volume, affine


# Global converter instance
_converter = None

def get_converter() -> DicomToNiftiConverter:
    """Get or create global converter instance."""
    global _converter
    if _converter is None:
        _converter = DicomToNiftiConverter()
    return _converter
