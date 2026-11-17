
# Brain Tumor Classification Using Hybrid ViT + GRU with Explainable AI

## Overview
An academic project that classifies brain MRI images into four categories using a Hybrid Vision Transformer and GRU model.

The application also integrates Explainable AI techniques to visualize model attribution and sensitivity.

## Classification Categories
- Glioma
- Meningioma
- No Tumor
- Pituitary

## Technologies Used
- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow
- ReportLab

## Explainable AI Methods
1. Integrated Gradients
2. Black-patch Occlusion Sensitivity
3. Blur-based Occlusion Sensitivity

## Features
- MRI image upload
- Brain tumor classification
- Class probability visualization
- XAI heatmap generation
- Original MRI and heatmap comparison
- Downloadable PDF report

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Disclaimer
This is an academic research prototype and is not a clinically validated diagnostic system. Predictions and heatmaps must not be used for medical decisions.
