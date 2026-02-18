#!/usr/bin/env python3
"""
Quick GPU setup verification script.
Tests GPU availability and provides installation instructions if needed.
"""

import sys
import subprocess

def check_nvidia_driver():
    """Check if NVIDIA driver is installed."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA driver detected")
            # Extract CUDA version
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    print(f"  CUDA Version: {cuda_version}")
                    return cuda_version
            return "unknown"
        return None
    except FileNotFoundError:
        return None

def check_cupy():
    """Check if CuPy is installed and working."""
    try:
        import cupy as cp
        print(f"✓ CuPy installed (version {cp.__version__})")
        
        # Test GPU access
        device = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"  Device: {props['name'].decode()}")
        print(f"  Compute Capability: {props['major']}.{props['minor']}")
        
        mem_info = device.mem_info
        total_mem = mem_info[1] / (1024**3)
        print(f"  VRAM: {total_mem:.1f} GB")
        
        return True
    except ImportError:
        print("✗ CuPy not installed")
        return False
    except Exception as e:
        print(f"✗ CuPy error: {e}")
        return False

def main():
    print("="*60)
    print("GPU Acceleration Setup Check")
    print("="*60)
    
    # Check NVIDIA driver
    cuda_version = check_nvidia_driver()
    
    if cuda_version is None:
        print("\n❌ NVIDIA driver not found!")
        print("\nPlease install NVIDIA drivers from:")
        print("https://www.nvidia.com/Download/index.aspx")
        sys.exit(1)
    
    # Check CuPy
    cupy_installed = check_cupy()
    
    if not cupy_installed:
        print("\n📦 CuPy Installation Instructions:")
        print("-"*60)
        
        if cuda_version and cuda_version.startswith('12'):
            package = "cupy-cuda12x"
        elif cuda_version and cuda_version.startswith('11'):
            package = "cupy-cuda11x"
        else:
            package = "cupy-cuda12x"
        
        print(f"Run: pip install {package}")
        print("\nOr add to your requirements.txt and run:")
        print("pip install -r requirements.txt")
        print("-"*60)
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ GPU acceleration is ready!")
    print("="*60)
    print("\nYour application will automatically use GPU acceleration.")
    print("Run: streamlit run src/app.py")
    
if __name__ == "__main__":
    main()
