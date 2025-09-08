# =========================
# Performance Configuration
# =========================
"""
Performance configuration for multi-threaded processing optimization.
Includes memory, threading, and I/O settings.
"""

import os
import multiprocessing
from typing import Dict, Any

class PerformanceConfig:
    """
    Class to manage application performance settings.
    """
    
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count() or 1
        self.memory_available = self._get_available_memory()
        
    def _get_available_memory(self) -> int:
        """
        Estimates available system memory.
        
        Returns:
            int: Available memory in MB
        """
        try:
            import psutil
            return psutil.virtual_memory().available // (1024 * 1024)
        except ImportError:
            # Fallback if psutil is not available
            return 4096  # Assume 4GB as default
    
    def get_optimal_workers(self, file_count: int = 1) -> int:
        """
        Calculates optimal number of workers based on system and workload.
        
        Args:
            file_count (int): Number of files to process
            
        Returns:
            int: Optimal number of workers
        """
        # CPU-based configuration
        cpu_based = min(8, self.cpu_count * 2)
        
        # Memory-based configuration (assuming ~500MB per worker)
        memory_based = max(1, self.memory_available // 500)
        
        # Workload-based configuration
        workload_based = min(file_count, 6)  # Maximum 6 parallel files
        
        # Return smallest value to avoid overload
        optimal = min(cpu_based, memory_based, workload_based)
        
        return max(1, optimal)  # Minimum 1 worker
    
    def get_chunk_size(self, total_slices: int) -> int:
        """
        Calculates optimal chunk size for slice processing.
        
        Args:
            total_slices (int): Total number of slices
            
        Returns:
            int: Optimal chunk size
        """
        if total_slices <= 20:
            return 5
        elif total_slices <= 50:
            return 10
        elif total_slices <= 100:
            return 15
        else:
            return 20
    
    def get_io_settings(self) -> Dict[str, Any]:
        """
        Returns optimized I/O settings.
        
        Returns:
            Dict: I/O settings
        """
        return {
            'compression_level': 1,  # Light compression for PNG
            'buffer_size': 8192,     # File reading buffer
            'concurrent_io': True,   # Concurrent I/O when possible
            'lazy_loading': True     # Lazy loading for large data
        }
    
    def get_memory_settings(self) -> Dict[str, Any]:
        """
        Returns memory management settings.
        
        Returns:
            Dict: Memory settings
        """
        return {
            'clear_cache_frequency': 10,  # Clear cache every 10 operations
            'gc_frequency': 5,            # Garbage collection every 5 files
            'max_cached_slices': 50,      # Maximum slices in cache
            'memory_limit_mb': self.memory_available * 0.7  # 70% of available memory
        }
    
    def should_use_threading(self, file_count: int) -> bool:
        """
        Determines if threading should be used based on workload.
        
        Args:
            file_count (int): Number of files
            
        Returns:
            bool: True if threading should be used
        """
        # Threading is beneficial for more than 1 file and multi-core systems
        return file_count > 1 and self.cpu_count > 1
    
    def get_progress_update_frequency(self, total_operations: int) -> int:
        """
        Calculates optimal frequency for progress updates.
        
        Args:
            total_operations (int): Total number of operations
            
        Returns:
            int: Update frequency (every N operations)
        """
        if total_operations <= 10:
            return 1
        elif total_operations <= 100:
            return 5
        else:
            return 10


# Global instance for configuration
performance_config = PerformanceConfig()


# =========================
# Performance Utilities
# =========================
def get_system_info() -> Dict[str, Any]:
    """
    Returns system information for performance debugging.
    
    Returns:
        Dict: System information
    """
    info = {
        'cpu_count': performance_config.cpu_count,
        'memory_available_mb': performance_config.memory_available,
        'optimal_workers': performance_config.get_optimal_workers(),
        'platform': os.name
    }
    
    try:
        import psutil
        info.update({
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent
        })
    except ImportError:
        info['psutil_available'] = False
    
    return info
