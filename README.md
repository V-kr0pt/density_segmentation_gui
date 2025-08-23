# Density Segmentation GUI

An interactive web application for medical image segmentation, specifically designed for density-based segmentation of NIfTI (Neuroimaging Informatics Technology Initiative) files. This tool provides an intuitive graphical interface for drawing masks, adjusting thresholds, processing medical imaging data, and leveraging AI-powered segmentation with SAM2.

## Features

- **Interactive Mask Drawing**: Draw custom masks directly on medical images using an intuitive canvas interface
- **Threshold Adjustment**: Fine-tune segmentation parameters with real-time visual feedback
- **AI-Powered Segmentation**: Advanced segmentation using SAM2 (Segment Anything Model 2) with automatic propagation
- **Batch Processing**: Process multiple NIfTI files efficiently with automated workflows
- **Visual Feedback**: Real-time preview of segmentation results across different slices
- **Output Management**: Organized output structure with comprehensive metadata and visualization

## Processing Modes

### 1. Single File Processing
Traditional workflow for processing one file at a time:
- Select one NIfTI file
- Draw custom mask using interactive canvas
- Adjust threshold parameters manually
- Process individual file

### 2. Batch Processing
Efficient processing of multiple files:
- Select multiple NIfTI files
- Draw masks for each file in sequence
- Set thresholds for each file
- Process all files simultaneously

### 3. SAM2 AI Processing ⭐
Advanced AI-powered segmentation workflow:
- Select NIfTI file and draw initial mask region
- Automatic threshold analysis and bounding box detection
- SAM2 inference on central slice with box prompts
- Video-like propagation through all slices
- Fully automated processing after initial mask drawing

## System Requirements

- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 4GB (8GB+ recommended for SAM2 processing)
- **GPU**: CUDA-compatible GPU recommended for SAM2 (8GB+ VRAM ideal)
- **Storage**: Sufficient space for input NIfTI files and generated outputs

## Installation

### Prerequisites

Ensure you have Python 3.10 or higher installed on your system. You can verify your Python version by running:

```bash
python --version
```

### Setting Up the Environment

#### 1. Clone the Repository

```bash
git clone https://github.com/V-kr0pt/density_segmentation_gui.git
cd density_segmentation_gui
```

#### 2. Create Virtual Environment

**For Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**For macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. SAM2 Setup (Optional - for AI Processing Mode)

For advanced SAM2 AI segmentation capabilities:

```bash
# Download SAM2 checkpoint
Follow the instructions in /checkpoints/README.md
```

This downloads the SAM2.1 Hiera Large model (~224MB) required for AI-powered segmentation.

## Usage

### Starting the Application

**For Windows:**
```cmd
# Ensure virtual environment is activated
venv\Scripts\activate
streamlit run src/app.py
```

**For macOS/Linux:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
streamlit run src/app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

### Workflow

The application offers three distinct processing modes accessible from the main menu:

#### Traditional Processing (Single/Batch)
1. **Draw Mask**: Place your NIfTI files (`.nii` format) in the `media/` directory, select a file, and use the interactive canvas to draw masks
2. **Adjust Threshold**: Use the threshold slider to fine-tune segmentation parameters with real-time preview
3. **Process**: Execute the complete segmentation pipeline and review generated outputs

#### SAM2 AI Processing 🤖
1. **Draw Initial Mask**: Select a NIfTI file and draw a basic mask region to guide the AI
2. **Automatic Analysis**: The system automatically detects optimal threshold values and generates bounding boxes
3. **AI Inference**: SAM2 performs initial segmentation on the central slice using the detected region
4. **Video-like Propagation**: SAM2 automatically propagates the segmentation through all slices, treating them as video frames
5. **Results**: Review comprehensive segmentation results across the entire volume

### Input/Output Structure

#### Input Directory
```
media/
├── sample_scan_1.nii
├── sample_scan_2.nii
└── ...
```

#### Output Directory
```
output/
├── scan_name/
│   ├── dense.nii              # Generated density mask
│   ├── mask.json              # Mask metadata
│   └── dense_mask/
│       ├── mask.nii           # Final processed mask
│       ├── slice_0_threshold_0.57.png
│       ├── slice_1_threshold_0.54.png
│       └── ...                # Individual slice visualizations
│
└── sam2_results/              # SAM2 AI processing outputs
    ├── sam2_propagated_mask.nii
    ├── sam2_propagation_report.txt
    └── slice_visualizations/
```

## Project Structure

```
density_segmentation_gui/
├── src/
│   ├── app.py                 # Main Streamlit application
│   ├── draw_step.py          # Mask drawing functionality
│   ├── threshold_step.py     # Threshold adjustment interface
│   ├── process_step.py       # Processing pipeline
│   └── utils.py              # Utility functions and classes
├── media/                    # Input NIfTI files directory
├── output/                   # Generated outputs directory
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
└── README.md                # Project documentation
```

## Dependencies

### Core Dependencies
- **Streamlit**: Web application framework
- **NiBabel**: NIfTI file handling
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **OpenCV**: Computer vision operations
- **Pillow**: Image processing
- **Streamlit-drawable-canvas**: Interactive drawing component

### SAM2 AI Processing (Optional)
- **PyTorch**: Deep learning framework
- **SAM2**: Segment Anything Model 2 for advanced AI segmentation
- **Supervision**: Computer vision utilities

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Medical image processing powered by [NiBabel](https://nipy.org/nibabel/)
- Interactive drawing capabilities provided by [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas)
- AI-powered segmentation using [SAM2](https://github.com/facebookresearch/segment-anything-2) (Meta AI Research)
