Write-Output "Installing PyTorch with CUDA 12.1 support..."

Write-Output "Removing existing PyTorch installations..."
pip uninstall torch torchvision torchaudio -y

Write-Output "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Output "Verifying CUDA installation..."
python -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print("Device count:", torch.cuda.device_count()); print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "N/A"); print("PyTorch version:", torch.__version__)'

Write-Output "Installation complete!"
Write-Output ""
Write-Output "Final verification:"
python -c 'import torch; print("✅ CUDA available:", torch.cuda.is_available()); print("✅ Device count:", torch.cuda.device_count())'


