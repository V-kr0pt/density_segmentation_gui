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
        # Find any slice with mask data
        mask_found = False
        valid_coords = None
        
        for slice_idx in range(mask_data.shape[0]):
            slice_mask = mask_data[slice_idx]
            coords = np.where(slice_mask > 0)
            
            if len(coords[0]) > 0 and len(coords[1]) > 0:
                valid_coords = coords
                mask_found = True
                break
        
        if not mask_found:
            return None, None, None, False
        
        # Calculate bounds safely
        y_coords, x_coords = valid_coords
        y_min = max(0, np.min(y_coords) - padding)
        y_max = min(image_data.shape[0], np.max(y_coords) + padding)
        x_min = max(0, np.min(x_coords) - padding) 
        x_max = min(image_data.shape[1], np.max(x_coords) + padding)
        
        # Validate bounds
        if y_min >= y_max or x_min >= x_max:
            return None, None, None, False
        
        roi_bounds = (x_min, y_min, x_max, y_max)
        
        # Extract ROI for all slices
        roi_slices = []
        roi_masks = []
        
        for slice_idx in range(image_data.shape[2]):
            # Extract ROI from image
            img_slice = image_data[:, :, slice_idx]
            roi_slice = img_slice[y_min:y_max, x_min:x_max]
            
            # Extract ROI from mask (handle size differences)
            mask_slice_idx = min(slice_idx, mask_data.shape[0] - 1)
            mask_slice = mask_data[mask_slice_idx]
            roi_mask = mask_slice[y_min:y_max, x_min:x_max]
            
            roi_slices.append(roi_slice)
            roi_masks.append(roi_mask)
        
        return roi_bounds, roi_slices, roi_masks, True
        
    except Exception as e:
        print(f"Error in ROI extraction: {e}")
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
    Apply threshold safely with proper error handling
    """
    try:
        if image.size == 0 or mask.size == 0:
            return np.array([], dtype=np.uint8)
        
        # Normalize image
        normalized = safe_normalize_for_sam2(image)
        
        # Apply threshold
        threshold_255 = threshold_value * 255
        thresholded = (normalized > threshold_255) & (mask > 0)
        
        # Return as uint8
        return thresholded.astype(np.uint8)
        
    except Exception as e:
        print(f"Error in threshold application: {e}")
        return np.zeros_like(image, dtype=np.uint8)
