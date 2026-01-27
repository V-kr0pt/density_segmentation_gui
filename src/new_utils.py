import numpy as np
import nibabel as nib
import re
import os
import cv2
import matplotlib.pyplot as plt


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
    
    Key principle: Keep images in native orientation, only transform for display,
    and properly inverse transform coordinates back to native space.
    """
    
    def __init__(self, padding=50):
        """
        Initialize display transformation tracker.
        
        Args:
            padding: padding pixels to add around image
        """
        self.padding = padding
        self.scale = 1.0
        self.padded_shape = None
        self.original_shape = None
        
    def prepare_for_display(self, image, max_width=1200, max_height=800):
        """
        Prepares image for display by padding, scaling, and rotating.
        TRACKS all transformations for later reversal.
        
        Args:
            image: 2D numpy array in NATIVE orientation
            max_width: maximum display width
            max_height: maximum display height
            
        Returns:
            PIL.Image: transformed image ready for canvas display
        """
        from PIL import Image
        
        self.original_shape = image.shape  # Store native shape
        
        # Step 1: Pad the image
        padded_image = np.pad(
            image,
            pad_width=((self.padding, self.padding), (self.padding, self.padding)),
            mode="constant",
            constant_values=0
        )
        self.padded_shape = padded_image.shape
        
        # Step 2: Normalize for display
        padded_image = ImageProcessor.normalize_image(padded_image)
        
        # Step 3: Calculate scale to fit in display area
        orig_height, orig_width = padded_image.shape
        self.scale = min(max_width / orig_width, max_height / orig_height, 1)
        
        # Step 4: Resize
        pil_height = int(padded_image.shape[0] * self.scale)
        pil_width = int(padded_image.shape[1] * self.scale)
        pil_image = Image.fromarray(padded_image).resize((pil_width, pil_height))
        
        # Step 5: Rotate 90 degrees for canvas display
        pil_image = pil_image.rotate(90, expand=True)
        
        return pil_image
    
    def canvas_to_native_coords(self, canvas_points, rotated_width, rotated_height):
        """
        Converts polygon points from canvas coordinates back to NATIVE image coordinates.
        
        This is the INVERSE of all display transformations:
        1. Undo rotation (90 degrees)
        2. Undo scaling
        3. Undo padding
        
        Args:
            canvas_points: list of (x, y) tuples from canvas
            rotated_width: width of rotated canvas
            rotated_height: height of rotated canvas
            
        Returns:
            list of (row, col) tuples in NATIVE image coordinates
        """
        native_points = []
        
        for x, y in canvas_points:
            # Step 1: Undo rotation (90 degrees clockwise)
            # After rotation: canvas(x, y) -> pre-rotation(rotated_height - y, x)
            px = rotated_height - y
            py = x
            
            # Step 2: Undo scaling
            px = px / self.scale
            py = py / self.scale
            
            # Step 3: Undo padding
            px -= self.padding
            py -= self.padding
            
            # Step 4: Clamp to valid image coordinates
            px = int(np.clip(px, 0, self.original_shape[0] - 1))
            py = int(np.clip(py, 0, self.original_shape[1] - 1))
            
            native_points.append((px, py))
        
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
        
        # Create mask in same orientation as image
        mask = np.zeros_like(image, dtype=np.uint8)
        pts = np.array([points], dtype=np.int32)
        mask = np.ascontiguousarray(mask)
        cv2.fillPoly(mask, pts, 255)
        
        # Apply mask to image
        result = cv2.bitwise_and(image, image, mask=mask)
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
        combined_mask = np.zeros_like(image, dtype=np.uint8)
        
        for poly in polygons:
            if len(poly) >= 3:
                _, mask = MaskManager.create_mask(image, poly)
                combined_mask = np.maximum(combined_mask, mask)  # Union
        
        result = cv2.bitwise_and(image, image, mask=combined_mask)
        return result, combined_mask
    
    @staticmethod
    def measure_mask_area(mask):
        """Return number of non-zero pixels in mask."""
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
    def save_mask(mask_2d, original_shape, affine, file_path, file_name="dense.nii"):
        """
        Saves a 2D mask as a 3D NIfTI file in NATIVE orientation.
        
        The key insight: The mask_2d is already in the NATIVE orientation of a slice.
        We just need to replicate it across all slices in the correct dimension.
        
        Args:
            mask_2d (numpy.ndarray): 2D binary mask in NATIVE slice orientation
            original_shape (tuple): 3D shape of the original volume (e.g., (512, 512, 50))
            affine (numpy.ndarray): 4x4 affine transformation matrix from original image
            file_path (str): Directory path where mask will be saved
            file_name (str): Output filename
        
        Returns:
            str: Full path to saved mask file
        """
        if len(original_shape) != 3:
            raise ValueError(f"original_shape must be 3D, got {len(original_shape)}D")
        
        os.makedirs(file_path, exist_ok=True)
        
        # Identify which dimension contains slices (usually smallest)
        slice_dim = np.argmin(original_shape)
        
        # Build the expected 2D shape for a slice
        other_dims = [i for i in range(3) if i != slice_dim]
        expected_slice_shape = (original_shape[other_dims[0]], original_shape[other_dims[1]])
        
        # Verify mask dimensions match expected slice dimensions
        # If they don't match, try transpose
        if mask_2d.shape != expected_slice_shape:
            if mask_2d.shape == expected_slice_shape[::-1]:
                mask_2d = mask_2d.T
            else:
                # Last resort: resize (should generally not happen with proper coordinate transform)
                print(f"WARNING: Mask shape {mask_2d.shape} doesn't match expected {expected_slice_shape}")
                mask_2d = cv2.resize(mask_2d, expected_slice_shape[::-1], interpolation=cv2.INTER_NEAREST)
        
        # Create 3D volume by replicating mask across all slices
        mask_3d = np.zeros(original_shape, dtype=np.uint8)
        
        if slice_dim == 0:
            # Slices along first dimension: shape = (n_slices, height, width)
            for i in range(original_shape[0]):
                mask_3d[i, :, :] = mask_2d
        elif slice_dim == 1:
            # Slices along second dimension: shape = (height, n_slices, width)
            for i in range(original_shape[1]):
                mask_3d[:, i, :] = mask_2d
        else:  # slice_dim == 2
            # Slices along third dimension: shape = (height, width, n_slices)
            for i in range(original_shape[2]):
                mask_3d[:, :, i] = mask_2d
        
        # Save with original affine - this preserves spatial registration
        mask_nii = nib.Nifti1Image(mask_3d, affine)
        full_path = os.path.join(file_path, file_name)
        nib.save(mask_nii, full_path)
        
        print(f"Mask saved to {full_path}")
        print(f"  Shape: {mask_3d.shape} (matches original: {original_shape})")
        print(f"  Slice dimension: {slice_dim}")
        
        return full_path
    
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
            binary_slice = (array > 0).astype(np.uint8)
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
        
        # Save NIfTI with original affine
        mask_nii = nib.Nifti1Image(volume, original_affine)
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