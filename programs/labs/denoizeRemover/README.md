# Image Denoising Application

A Python application for analyzing and removing noise from images using various filters.

## Features

- Interactive GUI for image processing
- Multiple denoising filters:
  - Median filter
  - Gaussian blur
  - Bilateral filter
  - Wiener filter
- Real-time noise analysis
- PSNR comparison
- Noise distribution visualization

## Requirements

- Python 3.7 or higher
- Required packages (install using `pip install -r requirements.txt`):
  - opencv-python
  - numpy
  - Pillow
  - matplotlib
  - scipy

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your images in the `photos` directory
2. Run the application:
   ```bash
   python main.py
   ```
3. Select an image from the file tree on the left
4. Choose a filter from the control panel
5. Click "Apply Filter" to process the image

## Interface

- Left panel: File system tree for image selection
- Right panel (divided into 4 quadrants):
  - Top left: Original image
  - Top right: Processed image
  - Bottom left: Noise distribution chart
  - Bottom right: PSNR comparison chart

## Supported Image Formats

- JPG/JPEG
- PNG
- BMP

## Notes

- The application automatically analyzes the type of noise in the image
- PSNR values are calculated to measure the effectiveness of the applied filter
- Images are automatically resized to fit the display area while maintaining aspect ratio 