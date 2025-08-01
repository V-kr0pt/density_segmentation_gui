# Density Segmentation GUI

A interactive web application for medical image segmentation, specifically designed for density-based segmentation of NIfTI (Neuroimaging Informatics Technology Initiative) files. This tool provides an  graphical interface for drawing masks, adjusting thresholds, and processing medical imaging data.

## Features

- **Interactive Mask Drawing**: Draw custom masks directly on medical images using an intuitive canvas interface
- **Threshold Adjustment**: Fine-tune segmentation parameters with real-time visual feedback
- **Batch Processing**: Process multiple NIfTI files efficiently with automated workflows
- **Visual Feedback**: Real-time preview of segmentation results across different slices
- **Output Management**: Organized output structure with comprehensive metadata and visualization

## System Requirements

- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: Minimum 4GB (8GB recommended for large datasets)
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

#### Step 1: Draw Mask
1. Place your NIfTI files (`.nii` format) in the `media/` directory
2. Select a file from the dropdown menu
3. Use the interactive canvas to draw masks on your medical images
4. Adjust drawing parameters as needed
5. Save your mask to proceed to the next step

#### Step 2: Adjust Threshold
1. Review the central slice of your original image and generated mask
2. Use the threshold slider to fine-tune segmentation parameters
3. Preview the thresholded result in real-time
4. Save the optimized threshold settings

#### Step 3: Process
1. Execute the complete segmentation pipeline
2. Monitor processing progress
3. Review generated outputs and visualizations

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

- **Streamlit**: Web application framework
- **NiBabel**: NIfTI file handling
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **OpenCV**: Computer vision operations
- **Pillow**: Image processing
- **Streamlit-drawable-canvas**: Interactive drawing component

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Medical image processing powered by [NiBabel](https://nipy.org/nibabel/)
- Interactive drawing capabilities provided by [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas)