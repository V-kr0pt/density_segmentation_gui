# Script to install PyTorch with CUDA support for SAM2

echo "Installing PyTorch with CUDA 12.1 support..."

# Check if CUDA is available
echo "🔍 Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA found: $(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')"
else
    echo "⚠️ Warning: nvcc not found. Make sure CUDA is properly installed."
fi

# Uninstall existing PyTorch versions
echo "📦 Removing existing PyTorch installations..."
pip uninstall torch torchvision torchaudio -y

# Install PyTorch with CUDA 12.1 using the correct index URL
echo "⚡ Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA installation
echo "🔍 Verifying CUDA installation..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('PyTorch version:', torch.__version__)"

echo "✅ Installation complete!"
echo ""
echo "🔍 Final verification:"
python -c "import torch; print('✅ CUDA available:', torch.cuda.is_available()); print('✅ Device count:', torch.cuda.device_count())"