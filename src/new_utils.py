import numpy as np
import nibabel as nib
import re
import os
import cv2
import matplotlib.pyplot as plt
import nibabel as nib


class ImageProcessor:
    """Image processing operations"""
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
    """Mask-specific operations"""
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
    def measure_mask_area(mask):
        ''' 
            Return mask non zeros
        '''
        return np.count_nonzero(mask)
    
    @staticmethod 
    def create_final_mask(folder_path, original_shape, original_affine):
        """
        Creates a 3D NIfTI mask from processed slice arrays.

        Args:
            folder_path: directory containing .npy slice files
            original_shape: 3D shape of original volume
            original_affine: affine matrix from original volume

        Returns:
            str: path to saved mask.nii file
        """
        
        mask_path = os.path.join(folder_path, 'mask.nii')

        # Load and sort NPY files
        npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        if not npy_files:
            raise FileNotFoundError(f"No NPY files found in {folder_path}")

        # Sort files numerically by slice index to preserve correct order
        def extract_slice_index(filename):
            """Extract slice index from filename like 'slice_0005_threshold_0.38.npy'"""
            match = re.search(r'slice_(\d+)', filename)
            return int(match.group(1)) if match else 0

        npy_files.sort(key=extract_slice_index)

        images = []
        for f in npy_files:
            array = np.load(os.path.join(folder_path, f))
            images.append(array)

        print(f"Loaded {len(images)} PNG slices")
        print(f"Original shape: {original_shape}")
        print(f"PNG image shape: {images[0].shape}")

        # Identify slice dimension in original volume (smallest dimension)
        slice_dim = np.argmin(original_shape)
        num_slices_original = original_shape[slice_dim]
        num_slices_png = len(images)

        assert num_slices_png == num_slices_original, "Something wrong happened: The number of .png files is diferent from the number of slices in original volume image."


        # Get spatial dimensions from original volume (the two non-slice dimensions)
        spatial_dims = [i for i in range(3) if i != slice_dim]
        original_spatial_shape = (original_shape[spatial_dims[0]], original_shape[spatial_dims[1]])

        print(f"Original spatial dimensions: {original_spatial_shape}")
        print(f"Numpy spatial dimensions: {images[0].shape}")

        # Stack with proper orientation
        volume = MaskManager._stack_arrays_preserving_orientation(images, original_shape)

        print(f"Final volume shape: {volume.shape}")
        print(f"Target shape: {original_shape}")
        
        # Convert to binary (0 and 1)
        volume_binary = (volume > 0).astype(np.uint8)

        # --- FIX: correct flipped orientation (Z-axis flip) ---
        volume_binary = np.flip(volume_binary, axis=2)
        
        # Save NIfTI
        mask_nii = nib.Nifti1Image(volume_binary, original_affine)
        nib.save(mask_nii, mask_path)        
        print(f"Mask successfully saved to {mask_path}")

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
        

    @staticmethod
    def _stack_arrays_preserving_orientation(arrays, original_shape):
        """
        Stacks 2D arrays into 3D volume while preserving original orientation.
        Assumes arrays are in the same orientation as original slices.
        """
        # Identify slice dimension
        slice_dim = np.argmin(original_shape)

        # Stack along the slice dimension
        if slice_dim == 0:
            volume = np.stack(arrays, axis=0)
        elif slice_dim == 1:
            volume = np.stack(arrays, axis=1)
        else:  # slice_dim == 2
            volume = np.stack(arrays, axis=2)

        # Verify dimensions match
        if volume.shape != original_shape:
            print(f"Warning: Stacked shape {volume.shape} doesn't match original {original_shape}")
            # Apply minimal transformations to match
            volume = MaskManager._match_volume_dimensions(volume, original_shape)

        return volume

    @staticmethod
    def _match_volume_dimensions(volume, target_shape):
        """
        Applies minimal transformations to match target dimensions.
        """
        # Simple transpose if dimensions are swapped
        if (volume.shape[0] == target_shape[1] and 
            volume.shape[1] == target_shape[0] and 
            volume.shape[2] == target_shape[2]):
            return np.transpose(volume, (1, 0, 2))
        elif (volume.shape[0] == target_shape[0] and 
              volume.shape[1] == target_shape[2] and 
              volume.shape[2] == target_shape[1]):
            return np.transpose(volume, (0, 2, 1))

        return volume  # Return as-is if no simple transformation works
    
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
        ax.text(0.5, 0.05, f'Threshold: {threshold:.2f}',
               ha='center', va='center',
               transform=ax.transAxes,
               color='white', fontsize=16)
        ax.axis('off')
        return fig
    
    @staticmethod
    def adjust_slice_threshold(image_slice, mask_slice, target_area):
        threshold = 0.8
        step = 0.01
        best_threshold = threshold
        best_diff = float('inf')

        while threshold >= 0:
            thresholded_image = ThresholdOperator.threshold_slice(image_slice, mask_slice, threshold)
            area = MaskManager.measure_mask_area(thresholded_image)
            diff = abs(area - target_area)

            if diff < best_diff:
                best_diff = diff
                best_threshold = threshold

            threshold -= step

        return best_threshold, ThresholdOperator.threshold_slice(image_slice, mask_slice, best_threshold)