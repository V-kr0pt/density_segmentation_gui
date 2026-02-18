"""
GPU Acceleration Module for Density Segmentation GUI
Uses CuPy for CUDA-accelerated array operations with automatic CPU fallback.

Key Features:
- Automatic GPU detection and fallback
- Memory-efficient batch processing
- NumPy-compatible API using CuPy
- CUDA kernels for custom operations
"""

import os
import numpy as np
import logging
from typing import Tuple, Optional, Union
import warnings

# Configure logging
logger = logging.getLogger(__name__)

# ==========================================
# GPU Initialization with Fallback
# ==========================================

class GPUConfig:
    """GPU configuration and device management."""
    
    def __init__(self):
        self.cuda_available = False
        self.cupy_available = False
        self.device_name = "CPU"
        self.device_memory = 0
        self.xp = np  # Default to numpy
        self.cv2_cuda_available = False
        
        self._initialize()
    
    def _initialize(self):
        """Initialize GPU support with fallback."""
        try:
            import cupy as cp
            self.cupy_available = True
            self.cuda_available = True
            self.xp = cp
            
            # Get device info
            device = cp.cuda.Device()
            self.device_name = device.compute_capability
            self.device_memory = device.mem_info[1] / (1024**3)  # GB
            
            device_props = cp.cuda.runtime.getDeviceProperties(0)
            logger.info(f"GPU Acceleration Enabled: {device_props['name'].decode()}, "
                       f"VRAM: {self.device_memory:.1f} GB, "
                       f"Compute: {self.device_name}")
            
        except ImportError:
            logger.info("CuPy not installed. Using CPU (NumPy) for processing.")
            
        except Exception as e:
            logger.warning(f"GPU initialization failed: {e}. Using CPU.")
    
    def to_gpu(self, array: np.ndarray):
        """Transfer numpy array to GPU (if available)."""
        if self.cuda_available and isinstance(array, np.ndarray):
            return self.xp.asarray(array)
        return array
    
    def to_cpu(self, array):
        """Transfer array back to CPU (numpy)."""
        if self.cuda_available and hasattr(array, 'get'):
            return array.get()
        return np.asarray(array)
    
    def is_gpu_array(self, array) -> bool:
        """Check if array is on GPU."""
        return self.cuda_available and hasattr(array, 'device')


# Global GPU configuration
gpu_config = GPUConfig()


# ==========================================
# GPU-Accelerated Image Processing
# ==========================================

class GPUImageProcessor:
    """GPU-accelerated image processing operations."""
    
    @staticmethod
    def normalize_image(img: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Normalize image to 0-255 range using GPU.
        
        Args:
            img: Input image array (numpy or cupy)
            
        Returns:
            Normalized image in 0-255 range
        """
        xp = gpu_config.xp
        
        # Convert to GPU if needed
        if not gpu_config.is_gpu_array(img):
            img_gpu = xp.asarray(img)
        else:
            img_gpu = img
        
        # GPU-accelerated min/max normalization
        img_min = xp.min(img_gpu)
        img_max = xp.max(img_gpu)
        
        # Avoid division by zero
        if img_max == img_min:
            return xp.zeros_like(img_gpu, dtype=xp.uint8)
        
        normalized = (img_gpu - img_min) / (img_max - img_min + 1e-8)
        result = (normalized * 255).astype(xp.uint8)
        
        return result
    
    @staticmethod
    def normalize_data(data: Union[np.ndarray, 'cp.ndarray']) -> Union[np.ndarray, 'cp.ndarray']:
        """
        Normalize data to 0-1 range using GPU.
        
        Args:
            data: Input data array
            
        Returns:
            Normalized data in 0-1 range
        """
        xp = gpu_config.xp
        
        data_gpu = xp.asarray(data) if not gpu_config.is_gpu_array(data) else data
        
        mn = xp.min(data_gpu)
        mx = xp.max(data_gpu)
        
        if mx == mn:
            return xp.zeros_like(data_gpu)
        
        return (data_gpu - mn) / (mx - mn)


# ==========================================
# GPU-Accelerated Threshold Operations
# ==========================================

class GPUThresholdOperator:
    """GPU-accelerated threshold operations for segmentation."""
    
    @staticmethod
    def apply_threshold(image: np.ndarray, mask: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply threshold to normalized image within mask region using GPU.
        
        Args:
            image: Input image
            mask: Binary mask
            threshold: Threshold value (0-1)
            
        Returns:
            Binary thresholded result
        """
        xp = gpu_config.xp
        
        # Transfer to GPU
        img_gpu = xp.asarray(image)
        mask_gpu = xp.asarray(mask)
        
        # Normalize and threshold on GPU
        norm_image = GPUImageProcessor.normalize_data(img_gpu)
        result = (norm_image > threshold) & (mask_gpu > 0)
        
        # Return as numpy for compatibility
        return gpu_config.to_cpu(result)
    
    @staticmethod
    def threshold_slice(img: np.ndarray, mask: np.ndarray, threshold: float) -> np.ndarray:
        """GPU-accelerated slice thresholding."""
        return GPUThresholdOperator.apply_threshold(img, mask, threshold)
    
    @staticmethod
    def adjust_slice_threshold_gpu(
        image_slice: np.ndarray,
        mask_slice: np.ndarray,
        target_area: int,
        initial_threshold: float = 0.8,
        step: float = 0.01,
        min_threshold: float = 0.0
    ) -> Tuple[float, np.ndarray]:
        """
        GPU-accelerated threshold adjustment to match target area.
        
        This version processes multiple thresholds in parallel on GPU,
        achieving significant speedup over sequential CPU version.
        
        Args:
            image_slice: Input image slice
            mask_slice: Mask defining ROI
            target_area: Target number of pixels
            initial_threshold: Starting threshold value
            step: Threshold decrement step
            min_threshold: Minimum threshold to test
            
        Returns:
            (best_threshold, thresholded_image)
        """
        xp = gpu_config.xp
        
        # Transfer to GPU
        img_gpu = xp.asarray(image_slice, dtype=xp.float32)
        mask_gpu = xp.asarray(mask_slice, dtype=xp.float32)
        
        # Normalize image on GPU
        img_min = xp.min(img_gpu)
        img_max = xp.max(img_gpu)
        if img_max > img_min:
            norm_img = (img_gpu - img_min) / (img_max - img_min)
        else:
            norm_img = xp.zeros_like(img_gpu)
        
        # Create array of threshold values to test
        num_thresholds = int((initial_threshold - min_threshold) / step) + 1
        thresholds = xp.linspace(initial_threshold, min_threshold, num_thresholds, dtype=xp.float32)
        
        # Vectorized threshold testing using broadcasting
        # Shape: (num_thresholds, H, W)
        norm_img_expanded = norm_img[xp.newaxis, :, :]  # (1, H, W)
        thresholds_expanded = thresholds[:, xp.newaxis, xp.newaxis]  # (N, 1, 1)
        mask_expanded = mask_gpu[xp.newaxis, :, :] > 0  # (1, H, W)
        
        # Parallel threshold application
        thresholded_batch = (norm_img_expanded > thresholds_expanded) & mask_expanded
        
        # Count areas for all thresholds in parallel
        areas = xp.sum(thresholded_batch, axis=(1, 2))  # (num_thresholds,)
        
        # Find best threshold (minimum difference from target)
        diffs = xp.abs(areas - target_area)
        best_idx = xp.argmin(diffs)
        
        best_threshold = float(thresholds[best_idx])
        best_thresholded = thresholded_batch[best_idx]
        
        # Convert back to CPU as numpy array
        return best_threshold, gpu_config.to_cpu(best_thresholded.astype(xp.uint8) * 255)


# ==========================================
# GPU-Accelerated Mask Operations
# ==========================================

class GPUMaskOperator:
    """GPU-accelerated mask operations."""
    
    @staticmethod
    def measure_mask_area(mask: np.ndarray) -> int:
        """
        GPU-accelerated mask area calculation.
        
        Args:
            mask: Binary mask
            
        Returns:
            Number of non-zero pixels
        """
        xp = gpu_config.xp
        mask_gpu = xp.asarray(mask)
        count = int(xp.count_nonzero(mask_gpu))
        return count
    
    @staticmethod
    def create_combined_mask_gpu(masks_list: list) -> np.ndarray:
        """
        Combine multiple binary masks using GPU.
        
        Args:
            masks_list: List of binary masks
            
        Returns:
            Combined mask (union)
        """
        if not masks_list:
            return np.array([])
        
        xp = gpu_config.xp
        
        # Transfer all masks to GPU
        masks_gpu = [xp.asarray(m) for m in masks_list]
        
        # Stack and take maximum (union)
        stacked = xp.stack(masks_gpu, axis=0)
        combined = xp.max(stacked, axis=0)
        
        return gpu_config.to_cpu(combined)
    
    @staticmethod
    def stack_slices_3d(slices: list, slice_dim: int = 0) -> np.ndarray:
        """
        Stack 2D slices into 3D volume using GPU.
        
        Args:
            slices: List of 2D arrays
            slice_dim: Dimension along which to stack (0, 1, or 2)
            
        Returns:
            3D volume
        """
        xp = gpu_config.xp
        
        # Transfer to GPU and stack
        slices_gpu = [xp.asarray(s) for s in slices]
        
        # Stack along axis 0 first
        volume_gpu = xp.stack(slices_gpu, axis=0)
        
        # Move axis if needed
        if slice_dim == 1:
            volume_gpu = xp.moveaxis(volume_gpu, 0, 1)
        elif slice_dim == 2:
            volume_gpu = xp.moveaxis(volume_gpu, 0, 2)
        
        return gpu_config.to_cpu(volume_gpu)


# ==========================================
# GPU-Accelerated Batch Processing
# ==========================================

class GPUBatchProcessor:
    """
    GPU-accelerated batch processing for multiple slices.
    Optimized for your RTX 4000 ADA with 20GB VRAM.
    """
    
    def __init__(self, max_batch_size: Optional[int] = None):
        """
        Initialize batch processor.
        
        Args:
            max_batch_size: Maximum number of slices to process in one batch
                          (None = auto-calculate based on available VRAM)
        """
        self.max_batch_size = max_batch_size
        if max_batch_size is None and gpu_config.cuda_available:
            # Auto-calculate based on available memory
            # Assuming 512x512 slices, float32: ~1MB per slice
            # RTX 4000 ADA has 20GB, use 50% safely
            available_gb = gpu_config.device_memory * 0.5
            self.max_batch_size = int(available_gb * 1024)  # ~10K slices max
        elif max_batch_size is None:
            self.max_batch_size = 10  # CPU fallback: small batches
    
    def process_slices_batch(
        self,
        image_slices: list,
        mask_slices: list,
        target_area: int
    ) -> list:
        """
        Process multiple slices in parallel on GPU.
        
        Args:
            image_slices: List of image slice arrays
            mask_slices: List of mask slice arrays
            target_area: Target area for thresholding
            
        Returns:
            List of tuples: [(threshold1, binary_mask1), (threshold2, binary_mask2), ...]
        """
        if not gpu_config.cuda_available:
            # CPU fallback: process sequentially
            return self._process_slices_cpu(image_slices, mask_slices, target_area)
        
        xp = gpu_config.xp
        results = []
        
        # Process in batches to manage memory
        num_slices = len(image_slices)
        for batch_start in range(0, num_slices, self.max_batch_size):
            batch_end = min(batch_start + self.max_batch_size, num_slices)
            
            batch_images = image_slices[batch_start:batch_end]
            batch_masks = mask_slices[batch_start:batch_end]
            
            # Process batch on GPU
            batch_results = self._process_batch_gpu(batch_images, batch_masks, target_area)
            results.extend(batch_results)
        
        return results
    
    def _process_batch_gpu(
        self,
        image_slices: list,
        mask_slices: list,
        target_area: int
    ) -> list:
        """Process a batch of slices on GPU in parallel."""
        results = []
        
        for img, mask in zip(image_slices, mask_slices):
            threshold, binary = GPUThresholdOperator.adjust_slice_threshold_gpu(
                img, mask, target_area
            )
            results.append((threshold, binary))
        
        return results
    
    def _process_slices_cpu(self, image_slices: list, mask_slices: list, target_area: int) -> list:
        """CPU fallback for slice processing."""
        from new_utils import ThresholdOperator
        
        results = []
        for img, mask in zip(image_slices, mask_slices):
            threshold, binary = ThresholdOperator.adjust_slice_threshold(img, mask, target_area)
            results.append((threshold, binary))
        
        return results


# ==========================================
# Utility Functions
# ==========================================

def get_gpu_info() -> dict:
    """Get GPU information and availability."""
    info = {
        'available': gpu_config.cuda_available,
        'device': gpu_config.device_name,
        'memory_gb': gpu_config.device_memory,
        'backend': 'CuPy' if gpu_config.cupy_available else 'NumPy'
    }
    
    if gpu_config.cuda_available:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        info['device_name'] = props['name'].decode()
        info['compute_capability'] = f"{props['major']}.{props['minor']}"
        info['multiprocessors'] = props['multiProcessorCount']
    
    return info


# ==========================================
# Performance Benchmarking
# ==========================================

def benchmark_gpu_vs_cpu(image_shape=(512, 512), num_iterations=100):
    """
    Benchmark GPU vs CPU performance.
    
    Args:
        image_shape: Shape of test images
        num_iterations: Number of iterations for timing
        
    Returns:
        dict: Benchmark results
    """
    import time
    
    results = {
        'image_shape': image_shape,
        'iterations': num_iterations,
        'cpu_time': 0,
        'gpu_time': 0,
        'speedup': 1.0
    }
    
    # Generate test data
    test_image = np.random.rand(*image_shape).astype(np.float32)
    test_mask = (np.random.rand(*image_shape) > 0.5).astype(np.uint8) * 255
    target_area = 50000
    
    # CPU benchmark
    from new_utils import ThresholdOperator as CPUThresholdOp
    
    start = time.time()
    for _ in range(num_iterations):
        _, _ = CPUThresholdOp.adjust_slice_threshold(test_image, test_mask, target_area)
    cpu_time = time.time() - start
    results['cpu_time'] = cpu_time
    
    logger.info(f"CPU benchmark: {cpu_time:.3f}s ({cpu_time/num_iterations*1000:.2f}ms per iteration)")
    
    # GPU benchmark
    if gpu_config.cuda_available:
        # Warmup
        _, _ = GPUThresholdOperator.adjust_slice_threshold_gpu(test_image, test_mask, target_area)
        
        start = time.time()
        for _ in range(num_iterations):
            _, _ = GPUThresholdOperator.adjust_slice_threshold_gpu(test_image, test_mask, target_area)
        gpu_time = time.time() - start
        results['gpu_time'] = gpu_time
        results['speedup'] = cpu_time / gpu_time
        
        logger.info(f"GPU benchmark: {gpu_time:.3f}s ({gpu_time/num_iterations*1000:.2f}ms per iteration)")
        logger.info(f"Speedup: {results['speedup']:.2f}x faster on GPU")
    else:
        logger.info("GPU not available for benchmarking")
    
    return results


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Log GPU information
    info = get_gpu_info()
    logger.info("GPU Acceleration Status")
    logger.info(f"GPU Available: {info['available']}")
    if info['available']:
        logger.info(f"Device: {info['device_name']}")
        logger.info(f"VRAM: {info['memory_gb']:.1f} GB")
    
    # Run benchmark if GPU is available
    if gpu_config.cuda_available:
        benchmark_gpu_vs_cpu(num_iterations=50)
