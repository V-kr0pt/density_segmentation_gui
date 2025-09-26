import numpy as np
import nibabel as nib
import os
import cv2
from ImageLoader import BaseImageLoader
import matplotlib.pyplot as plt
import nibabel as nib

class OrientationConfig:
    """Define regras consistentes para orientação da imagem"""
    
    STANDARD_NIFTI = {
        'rotation': 90,
        'flip_axes': [0],  # flip vertical
        'k': 1
    }
    
    STANDARD_DICOM = {
        'rotation': -90,
        'flip_axes': [0, 1],  # flip vertical e horizontal
        'k': -1
    }

class ImageProcessor:
    """Operações de processamento de imagem"""
    # Normalização, filtros, etc.
    @staticmethod
    def normalize_image(img):
        """
            Return the image in 0-255 range
        """
        img = img - np.min(img)
        img = img / (np.max(img) + 1e-8)
        return (img * 255).astype(np.uint8)
    
    @staticmethod
    def normalize_data(data):
        """
            Return data in 0-1 range
        """
        mn, mx = data.min(), data.max()
        return (data - mn) / (mx - mn)

class MaskManager:
    """Operações específicas de máscara"""
    # Criação, manipulação, salvamento de máscaras
    @staticmethod
    def create_mask(image, points, reduction_scale=1.0):
        """
        Creates a binary mask from a polygon defined by points and applies it to an image.

        This method generates a polygon mask based on the provided points, scales them 
        according to the reduction scale, and returns both the masked image and the binary mask.
        Useful for single-region segmentation tasks.

        Args:
            image (numpy.ndarray): Input image to apply the mask to. Can be grayscale or RGB.
            points (list of tuples): Polygon coordinates as [(x1,y1), (x2,y2), ...] 
                                    in reduced scale coordinates.
            reduction_scale (float, optional): Scale factor used to reduce the original image. 
                                             Points will be scaled up by 1/reduction_scale.
                                             Defaults to 1.0 (no scaling).

        Returns:
            tuple: A tuple containing:
                - masked_image (numpy.ndarray): Original image with mask applied 
                                              (areas outside mask are blackened).
                - binary_mask (numpy.ndarray): Binary mask where polygon area is white (255) 
                                             and background is black (0).

        Raises:
            ValueError: If fewer than 3 points are provided (insufficient for a polygon).

        Example:
            -> points = [(100, 100), (150, 100), (125, 150)]
            -> masked_img, mask = MaskManager.create_mask(image, points, reduction_scale=0.5)
            -> print(mask.shape)  # Same as image shape
            -> print(np.unique(mask))  # [0, 255]
        """
        
        if len(points) < 3:
            raise ValueError("At least 3 points are required to create a polygon mask.")
        
        # Adjust the points to the original image scale
        scale = 1 / reduction_scale
        scaled_points = [(int(x * scale), int(y * scale)) for (x, y) in points]
        
        mask = np.zeros_like(image, dtype=np.uint8)
        pts = np.array([scaled_points], dtype=np.int32)
        mask = np.ascontiguousarray(mask)
        cv2.fillPoly(mask, pts, 255)

        # Use only one channel for bitwise operations 
        result = cv2.bitwise_and(image, image, mask=mask)
        return result, mask
    
    @staticmethod
    def create_combined_mask(image, polygons, reduction_scale=1.0):
        """
        Creates a combined binary mask from multiple polygons and applies it to an image.

        This method generates a unified mask by combining multiple polygon regions through
        union operation. Each polygon contributes to the final mask, allowing multi-region
        segmentation in a single operation.

        Args:
            image (numpy.ndarray): Input image to apply the combined mask to.
            polygons (list of lists): List of polygons, where each polygon is a list of 
                                     coordinate tuples: [[(x1,y1), (x2,y2), ...], ...].
            reduction_scale (float, optional): Scale factor used to reduce the original image.
                                             Points will be scaled up by 1/reduction_scale.
                                             Defaults to 1.0 (no scaling).

        Returns:
            tuple: A tuple containing:
                - masked_image (numpy.ndarray): Original image with combined mask applied.
                - combined_mask (numpy.ndarray): Unified binary mask where all polygon areas 
                                               are white (255) and background is black (0).

        Note:
            Polygons with fewer than 3 points are automatically skipped as they cannot form 
            valid polygons. The method uses maximum operation to combine masks, ensuring 
            overlapping regions are properly handled.

        Example:
            >> polygons = [
            ..     [(100, 100), (150, 100), (125, 150)],  # Triangle
            ..     [(200, 200), (250, 200), (225, 250)]   # Another triangle
            .. ]
            >> masked_img, combined_mask = MaskManager.create_combined_mask(
            ..     image, polygons, reduction_scale=0.5
            .. )
            >> # Both regions will be visible in the combined mask
        """

        combined_mask = np.zeros_like(image, dtype=np.uint8)
        for poly in polygons:
            if len(poly) >= 3:
                _, mask = MaskManager.create_mask(image, poly, reduction_scale=reduction_scale)
                combined_mask = np.maximum(combined_mask, mask)  # union of all polygons
        result = cv2.bitwise_and(image, image, mask=combined_mask)
        return result, combined_mask
    
    
    @staticmethod
    def save_mask(mask, original_shape, nb_of_slices, affine, file_path, file_name="dense.nii"):
        """
        Saves a 2D mask as a 3D NIfTI file while preserving the original volume dimensions.

        This method ensures the output mask has exactly the same dimensions as the original
        medical image, which is essential for proper registration and analysis in medical
        software. The mask is placed in the correct slice position within the 3D volume.

        Args:
            mask (numpy.ndarray): 2D binary mask to be saved (single slice).
            original_shape (tuple): 3D shape of the original image (slices, height, width).
            affine (numpy.ndarray): 4x4 affine transformation matrix for NIfTI header.
            file_path (str): Directory path where the mask will be saved.
            file_name (str, optional): Output filename. Defaults to "dense.nii".

        Returns:
            str: Full path to the saved mask file.

        Raises:
            ValueError: If original_shape is not 3D or incompatible with mask dimensions.

        Example:
            >>> # Original image shape: (50, 512, 512) - 50 slices of 512x512
            >>> # Mask is for slice 25 (central slice)
            >>> save_mask(mask_2d, (50, 512, 512), affine, "/output/path")
            >>> # Result: 3D volume with mask only in slice 25, others are zeros
        """
        if len(original_shape) != 3:
            raise ValueError(f"original_shape must be 3D, got {len(original_shape)}D")

        # Verify mask dimensions match the original image's slice dimensions
        if nb_of_slices not in original_shape:
            raise ValueError(f"Wrong number of slices: it was passed {nb_of_slices} to the function but original shape was {original_shape}")

        os.makedirs(file_path, exist_ok=True)

        # Ensure the mask is properly oriented for the original volume
        # Apply the same transformations that were used when extracting the slice
        oriented_mask = MaskManager._orient_mask_for_volume(mask, original_shape)

        # Apply the same affine transformation as the original image
        mask_nii = nib.Nifti1Image(oriented_mask, affine)

        full_path = os.path.join(file_path, file_name)
        nib.save(mask_nii, full_path)

        print(f"Mask saved to {full_path} with original dimensions: {mask_nii.shape}")
        return full_path

    @staticmethod
    def _orient_mask_for_volume(mask_2d, original_shape):
        """
        Creates a 3D volume from a 2D mask that matches the original volume's dimension order.
        
        Args:
            mask_2d (numpy.ndarray): 2D binary mask from canvas drawing
            original_shape (tuple): 3D shape of original volume (dim1, dim2, dim3)
            
        Returns:
            numpy.ndarray: 3D volume with mask replicated across slices, matching original_shape
        """
        # First, ensure the 2D mask matches the slice dimensions of the original volume
        slice_dims = MaskManager._get_slice_dimensions(original_shape)
        oriented_mask_2d = MaskManager._orient_2d_mask(mask_2d, slice_dims)
        print(f"slices_dims:{slice_dims}")
        
        # Create 3D volume by replicating the mask across all slices
        # Determine which dimension contains the slices
        slice_dim = MaskManager._identify_slice_dimension(original_shape)
        
        # Create empty volume with original shape
        mask_3d = np.zeros(original_shape, dtype=np.uint8)
        print(original_shape)
        
        # Use np.repeat for better performance
        if slice_dim == 0:  # Slices are first dimension (z, x, y)
            # Repeat along the first axis (axis=0)
            mask_3d = np.repeat(oriented_mask_2d[np.newaxis, :, :], original_shape[0], axis=0)

        elif slice_dim == 1:  # Slices are second dimension (x, z, y)
            # Repeat along the second axis (axis=1)
            mask_3d = np.repeat(oriented_mask_2d[:, np.newaxis, :], original_shape[1], axis=1)

        else:  # Slices are third dimension (x, y, z)
            # Repeat along the third axis (axis=2)
            mask_3d = np.repeat(oriented_mask_2d[:, :, np.newaxis], original_shape[2], axis=2)

        return mask_3d

    @staticmethod
    def _get_slice_dimensions(original_shape):
        """
        Identifies which dimensions correspond to slice height and width.
        Returns (height, width) of individual slices.
        """
        # Find the slice dimension (smallest dimension)
        slice_dim = np.argmin(original_shape)
        
        # The other two dimensions are the slice height and width
        other_dims = [i for i in range(3) if i != slice_dim]
        
        # Return the dimensions in their original order
        height_dim, width_dim = other_dims[0], other_dims[1]
        
        print(f"Slice dimension: {slice_dim} (size: {original_shape[slice_dim]})")
        
        return (original_shape[height_dim], original_shape[width_dim])

    @staticmethod
    def _identify_slice_dimension(original_shape):
        """
        Identifies which dimension corresponds to the slice direction (usually the smallest dimension).
        """
        return np.argmin(original_shape)

    @staticmethod
    def _orient_2d_mask(mask_2d, target_slice_dims):
        """
        Orients the 2D mask to match the target slice dimensions (height, width).
        """
        target_height, target_width = target_slice_dims

        # If dimensions already match, return as-is
        if mask_2d.shape == (target_height, target_width):
            return mask_2d

        # If dimensions are swapped, transpose
        if mask_2d.shape == (target_width, target_height):
            return mask_2d.T

        # If still no match, resize to fit
        return cv2.resize(mask_2d.astype(np.float32), (target_width, target_height)).astype(np.uint8)

class ThresholdOperator:
    @staticmethod
    def apply_threshold(image, mask, threshold):
        norm_image = ImageProcessor.normalize_data(image)
        return (norm_image > threshold) & (mask > 0)

    @staticmethod
    def threshold_slice(img, mask, threshold):
        bin_mask = ThresholdOperator.apply_threshold(img, mask, threshold)
        return bin_mask
    
    @staticmethod
    def display_thresholded_slice(img, mask, threshold):
        bin_mask = ThresholdOperator.threshold_slice(img, mask, threshold)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img, cmap='gray')
        ax.imshow(bin_mask, cmap='jet', alpha=0.2)
        ax.text(0.5, 0.05, f'Threshold: {threshold:.3f}',
               ha='center', va='center',
               transform=ax.transAxes,
               color='white', fontsize=16)
        ax.axis('off')
        return fig