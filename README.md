# MNIST Digit Classification Pipeline

A Python pipeline for classifying handwritten digits using a trained scikit-learn model.

## 📋 Overview

This project provides a simple and efficient pipeline to classify handwritten digit images (0-9) using machine learning. The pipeline handles image preprocessing, including contrast enhancement, inversion, cropping, and resizing to prepare images for prediction.

## 🚀 Features

- ✅ Automatic image preprocessing (grayscale conversion, contrast enhancement, inversion)
- ✅ Smart digit detection with bounding box cropping
- ✅ Aspect ratio preservation during resizing

## 📦 Requirements

```bash
pip install numpy pillow scikit-learn joblib
```

### Dependencies:
- `numpy` - Array operations
- `pillow` (PIL) - Image processing
- `scikit-learn` - Machine learning models
- `joblib` - Model serialization



## 🎯 How It Works

The pipeline performs the following preprocessing steps:

1. **Load & Convert** - Opens the image and converts to grayscale
2. **Contrast Enhancement** - Binarizes the image (threshold: 150)
3. **Smart Inversion** - Inverts colors if background is white (ensures white digits on black background)
4. **Bounding Box Crop** - Removes excess whitespace around the digit
5. **Resize** - Scales to 20×20 while preserving aspect ratio
6. **Canvas Placement** - Centers the digit on a 28×28 black canvas
7. **Normalization** - Scales pixel values to [0, 1]
8. **Flattening** - Converts to 1D array (784 features) for sklearn


## 📁 Project Structure

```
MNIST-Classification-Pipeline/
├── digital_pipeline.py      # Main pipeline class
├── model.joblib             # Your trained model
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── testprediction.py       #Runs the pipeline with the desired model and image
└── examples/               # Example images (optional)

```


### Image Requirements
- **Format:** PNG, JPG, JPEG, or any PIL-supported format
- **Content:** Single handwritten digit (0-9)
- **Background:** Works with both white and black backgrounds (auto-inverts)
- **Quality:** Clear, well-contrasted digits work best


### Import errors?
```bash
pip install --upgrade numpy pillow scikit-learn joblib
```
### Extra Note 
Git ls has been used here as model size is above 100MB
