# SAM2 Mode Implementation

## Overview
This implementation adds a new SAM2 mode to the Density Segmentation GUI that uses Meta's Segment Anything Model 2 (SAM2) for advanced video segmentation propagation.

## How SAM2 Mode Works

### 1. **First Frame Processing**
- The user draws a region of interest (same as traditional mode)
- A threshold is applied to the first slice in the drawn region
- SAM2 Image Predictor performs inference on this thresholded region
- This creates a high-quality initial segmentation mask

### 2. **Video Propagation**
- All slices from the drawn region are assembled into a video sequence
- The first frame uses the thresholded + SAM2 segmented version
- Subsequent frames use the original image data (no threshold)
- SAM2 Video Predictor propagates the segmentation through all frames
- This maintains temporal consistency across slices

### 3. **Output**
- Individual masks for each slice are saved as PNG files
- A combined visualization showing all slices with overlaid masks
- JSON results file with processing status

## Differences from Traditional Mode

| Aspect | Traditional Mode | SAM2 Mode |
|--------|-----------------|-----------|
| **First Frame** | Threshold only | Threshold + SAM2 inference |
| **Subsequent Frames** | Dynamic thresholding | SAM2 video propagation |
| **Consistency** | Parameter-based | AI-driven temporal consistency |
| **Dependencies** | Basic Python libs | SAM2 + PyTorch + CUDA |
| **Processing Time** | Fast | Slower (AI inference) |
| **Quality** | Good for uniform density | Better for complex shapes |

## Requirements

### Software Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Model Checkpoint
```bash
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### Hardware Recommendations
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- **RAM**: 16GB+ system memory
- **CPU**: Modern multi-core processor

## Files Modified/Added

### New Files
- `src/batch_sam2_process_step.py` - Main SAM2 processing implementation
- `README_SAM2.md` - This documentation file

### Modified Files
- `src/app.py` - Added mode selection and routing logic
- `src/batch_threshold_step.py` - Added mode-aware navigation
- `src/sam_utils.py` - Enhanced SAM2 utilities (already existed)

## Usage Flow

1. **Mode Selection**: User chooses between Traditional and SAM2 modes
2. **File Selection**: Same as traditional mode
3. **Draw Step**: Same as traditional mode - user draws regions of interest
4. **Threshold Step**: Same as traditional mode - user sets thresholds
5. **SAM2 Processing**: 
   - Loads SAM2 models
   - Processes first frame with threshold + SAM2
   - Propagates segmentation through remaining frames
   - Saves results and visualizations

## Technical Implementation Details

### SAM2VideoManager Class
- Manages SAM2 video predictor model loading
- Handles inference state initialization
- Provides propagation functionality

### Video Frame Preparation
- Converts NIfTI slices to RGB format for SAM2
- Normalizes intensity values to 0-255 range
- Applies threshold only to first frame

### Mask Propagation
- Uses SAM2's built-in video propagation
- Maintains object IDs across frames
- Returns per-frame segmentation masks

### Output Generation
- Saves individual mask files per slice
- Creates combined visualization
- Generates processing report

## Error Handling

The implementation includes comprehensive error handling for:
- Missing SAM2 dependencies
- Model checkpoint availability
- CUDA/GPU availability issues
- File processing errors
- Memory allocation problems

## Performance Considerations

- **Memory Usage**: SAM2 requires significant GPU memory
- **Processing Time**: Varies based on region size and number of slices
- **Batch Size**: Currently processes one file at a time
- **Optimization**: Could be improved with batching and memory management

## Future Enhancements

1. **Batch Processing**: Process multiple files simultaneously
2. **Custom Prompts**: Allow more sophisticated prompting strategies
3. **Interactive Refinement**: Let users refine masks interactively
4. **Model Selection**: Support different SAM2 model sizes
5. **Export Formats**: Additional output formats (DICOM, NIfTI masks)
