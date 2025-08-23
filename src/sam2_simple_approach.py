"""
Simplified SAM2 approach that focuses on robust mask creation and propagation
"""
import numpy as np
import cv2
from PIL import Image

def create_simple_initial_mask(thresholded_region, original_mask_region, min_area=50):
    """
    Create a simple but robust initial mask from thresholded data
    
    Args:
        thresholded_region: Result of threshold application
        original_mask_region: Original drawn mask region
        min_area: Minimum area in pixels for a valid mask
    
    Returns:
        mask: Binary mask for SAM2, or None if failed
    """
    try:
        # Ensure inputs are valid
        if thresholded_region.size == 0 or original_mask_region.size == 0:
            return None
        
        # Convert to binary masks
        thresh_binary = (thresholded_region > 0).astype(np.uint8)
        mask_binary = (original_mask_region > 0).astype(np.uint8)
        
        # Combine masks
        combined_mask = thresh_binary & mask_binary
        
        # Check if we have enough area
        if np.sum(combined_mask) < min_area:
            # Fallback to original mask if threshold result is too small
            combined_mask = mask_binary
        
        # Final validation
        if np.sum(combined_mask) == 0:
            return None
        
        return combined_mask
        
    except Exception as e:
        print(f"Error creating initial mask: {e}")
        return None

def safe_roi_extraction(image_data, mask_data, padding=10):
    """
    Safely extract region of interest with robust bounds checking
    
    Args:
        image_data: 3D image data (H, W, D)
        mask_data: 3D mask data 
        padding: Padding around the ROI
    
    Returns:
        roi_bounds, roi_slices, roi_masks, success
    """
    try:
        print(f"ROI Extraction Debug:")
        print(f"  Image shape: {image_data.shape}")
        print(f"  Mask shape: {mask_data.shape}")
        print(f"  Image dtype: {image_data.dtype}")
        print(f"  Mask dtype: {mask_data.dtype}")
        
        # Validate input dimensions
        if len(image_data.shape) != 3:
            print(f"  Error: Image data must be 3D, got shape {image_data.shape}")
            return None, None, None, False
            
        if len(mask_data.shape) != 3:
            print(f"  Error: Mask data must be 3D, got shape {mask_data.shape}")
            return None, None, None, False
        
        # Find any slice with mask data
        mask_found = False
        valid_coords = None
        best_slice_idx = -1
        max_mask_area = 0
        
        # Search all slices to find the one with most mask data
        for slice_idx in range(mask_data.shape[0]):
            slice_mask = mask_data[slice_idx]
            coords = np.where(slice_mask > 0)
            
            if len(coords[0]) > 0 and len(coords[1]) > 0:
                mask_area = len(coords[0])
                if mask_area > max_mask_area:
                    max_mask_area = mask_area
                    valid_coords = coords
                    best_slice_idx = slice_idx
                    mask_found = True
        
        if not mask_found:
            print(f"  Error: No mask data found in any slice")
            return None, None, None, False
        
        print(f"  Best mask found in slice {best_slice_idx} with {max_mask_area} pixels")
        
        # Calculate bounds safely
        y_coords, x_coords = valid_coords
        
        # Get raw bounds
        raw_y_min, raw_y_max = np.min(y_coords), np.max(y_coords)
        raw_x_min, raw_x_max = np.min(x_coords), np.max(x_coords)
        
        print(f"  Raw bounds: y({raw_y_min},{raw_y_max}) x({raw_x_min},{raw_x_max})")
        
        # Apply padding and ensure bounds are within image dimensions
        y_min = max(0, raw_y_min - padding)
        y_max = min(image_data.shape[0], raw_y_max + padding + 1)  # +1 for inclusive indexing
        x_min = max(0, raw_x_min - padding)
        x_max = min(image_data.shape[1], raw_x_max + padding + 1)  # +1 for inclusive indexing
        
        print(f"  Padded bounds: y({y_min},{y_max}) x({x_min},{x_max})")
        print(f"  Image dimensions: H={image_data.shape[0]}, W={image_data.shape[1]}")
        
        # Validate bounds
        if y_min >= y_max or x_min >= x_max:
            print(f"  Error: Invalid bounds after padding")
            return None, None, None, False
        
        # Ensure minimum ROI size
        min_roi_size = 32  # Minimum 32x32 region
        roi_height = y_max - y_min
        roi_width = x_max - x_min
        
        if roi_height < min_roi_size or roi_width < min_roi_size:
            print(f"  Warning: ROI too small ({roi_width}x{roi_height}), expanding...")
            
            # Expand to minimum size
            center_y = (y_min + y_max) // 2
            center_x = (x_min + x_max) // 2
            
            half_size = min_roi_size // 2
            y_min = max(0, center_y - half_size)
            y_max = min(image_data.shape[0], center_y + half_size)
            x_min = max(0, center_x - half_size)
            x_max = min(image_data.shape[1], center_x + half_size)
            
            print(f"  Expanded bounds: y({y_min},{y_max}) x({x_min},{x_max})")
        
        roi_bounds = (x_min, y_min, x_max, y_max)
        
        # Extract ROI for all slices
        roi_slices = []
        roi_masks = []
        
        for slice_idx in range(image_data.shape[2]):
            try:
                # Extract ROI from image
                img_slice = image_data[:, :, slice_idx]
                roi_slice = img_slice[y_min:y_max, x_min:x_max]
                
                # Extract ROI from mask (handle size differences)
                mask_slice_idx = min(slice_idx, mask_data.shape[0] - 1)
                mask_slice = mask_data[mask_slice_idx]
                roi_mask = mask_slice[y_min:y_max, x_min:x_max]
                
                # Validate extracted ROIs
                if roi_slice.size == 0:
                    print(f"  Warning: Empty ROI slice at index {slice_idx}")
                    # Create a small dummy region if needed
                    roi_slice = np.zeros((min_roi_size, min_roi_size), dtype=image_data.dtype)
                
                if roi_mask.size == 0:
                    print(f"  Warning: Empty ROI mask at index {slice_idx}")
                    # Create a small dummy mask if needed
                    roi_mask = np.zeros((min_roi_size, min_roi_size), dtype=mask_data.dtype)
                
                roi_slices.append(roi_slice)
                roi_masks.append(roi_mask)
                
            except Exception as slice_error:
                print(f"  Error extracting slice {slice_idx}: {slice_error}")
                # Add dummy data to maintain slice count
                roi_slices.append(np.zeros((min_roi_size, min_roi_size), dtype=image_data.dtype))
                roi_masks.append(np.zeros((min_roi_size, min_roi_size), dtype=mask_data.dtype))
        
        print(f"  Successfully extracted {len(roi_slices)} ROI slices")
        print(f"  ROI slice dimensions: {roi_slices[0].shape if roi_slices else 'None'}")
        
        return roi_bounds, roi_slices, roi_masks, True
        
    except Exception as e:
        print(f"Error in ROI extraction: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, False

def safe_normalize_for_sam2(image):
    """
    Safely normalize image for SAM2 processing
    """
    try:
        if image.size == 0:
            return np.array([], dtype=np.uint8)
        
        # Handle edge cases
        if np.all(image == image.flat[0]):  # All values are the same
            return np.zeros_like(image, dtype=np.uint8)
        
        # Safe normalization
        img_min = np.min(image)
        img_max = np.max(image)
        
        if img_max > img_min:
            normalized = (image - img_min) / (img_max - img_min)
            return (normalized * 255).astype(np.uint8)
        else:
            return np.zeros_like(image, dtype=np.uint8)
            
    except Exception as e:
        print(f"Error in normalization: {e}")
        return np.zeros_like(image, dtype=np.uint8)

def apply_safe_threshold(image, mask, threshold_value):
    """
    Apply threshold safely using the EXACT same logic as batch_threshold_step
    This matches ThresholdOperations.threshold_image() exactly
    """
    try:
        if image.size == 0 or mask.size == 0:
            return np.array([], dtype=np.uint8)
        
        # Use the EXACT same normalization as ThresholdOperations.normalize_data
        mn, mx = image.min(), image.max()
        if mx > mn:
            norm = (image - mn) / (mx - mn)  # Exact same normalization
        else:
            norm = np.zeros_like(image)
        
        # Apply threshold using EXACT same logic as ThresholdOperations.threshold_image
        thresholded = (norm > threshold_value) & (mask > 0)
        
        # Return as uint8
        return thresholded.astype(np.uint8)
        
    except Exception as e:
        print(f"Error in threshold application: {e}")
        return np.zeros_like(image, dtype=np.uint8)
