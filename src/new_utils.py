import numpy as np
import nibabel as nib
import re
from PIL import Image
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
          1) pad
          2) normalize
          3) rotate (DISPLAY ONLY)
          4) scale to fit
        """
        

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

        mask = np.zeros_like(image, dtype=np.uint8)

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
        combined_mask = np.zeros_like(image, dtype=np.uint8)
        
        for poly in polygons:
            if len(poly) >= 3:
                _, mask = MaskManager.create_mask(image, poly)
                combined_mask = np.maximum(combined_mask, mask)  # Union
        
        result = image * (combined_mask > 0)
        #result = cv2.bitwise_and(image, image, mask=combined_mask)
        return result, combined_mask
    
    @staticmethod
    def measure_mask_area(mask):
        """Return number of non-zero pixels in mask."""
        return np.count_nonzero(mask)
           

    @staticmethod
    def save_mask(mask_2d, original_shape, affine, file_path, img_type, file_name="dense.nii"):
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

        # Ensure binary values (0/1) for saved mask
        mask_2d = (mask_2d > 0).astype(np.uint8)

        if img_type == "DICOM":
            print("Detected DICOM image!")
            #mask_2d = np.flipud(mask_2d)
            #mask_2d = np.rot90(mask_2d, k=1)

        
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
        
        #if img_type == "DICOM":
        #    mask_3d = mask_2d[np.newaxis, :, :].astype(np.uint8)
        #else:
            # Keep existing behavior for NIfTI / volumetric data
        slice_dim = int(np.argmin(original_shape))
        mask_3d = np.zeros(original_shape, dtype=np.uint8)

        if slice_dim == 0:
            for i in range(original_shape[0]):
                mask_3d[i, :, :] = mask_2d
        elif slice_dim == 1:
            for i in range(original_shape[1]):
                mask_3d[:, i, :] = mask_2d
        else:
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
