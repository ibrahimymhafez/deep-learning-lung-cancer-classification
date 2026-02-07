# Deep Learning-Based Multi-Class Lung Cancer Classification

A deep learning project for classifying lung cancer from histopathological images using Convolutional Neural Networks (CNN). This system distinguishes between normal lung tissue and two types of lung cancer with high accuracy.

## Project Overview

This project implements a CNN-based classifier to distinguish between:
- **Normal lung tissue** (lung_n)
- **Lung Adenocarcinoma** (lung_aca)
- **Lung Squamous Cell Carcinoma** (lung_scc)

## Dataset

The dataset consists of 15,000 histopathological images (5,000 per class) from the [Lung and Colon Cancer Histopathological Images dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) available on Kaggle.

## Project Structure

```
Lung Cancer Detection/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── config/
│   └── config.py               # Configuration parameters
├── data/
│   └── data_loader.py          # Dataset loading and extraction
├── models/
│   └── cnn_model.py            # CNN model architecture
├── preprocessing/
│   └── image_processor.py      # Image preprocessing utilities
├── utils/
│   ├── visualization.py        # Visualization functions
│   └── callbacks.py            # Custom training callbacks
├── training/
│   └── trainer.py              # Model training logic
├── evaluation/
│   └── evaluator.py            # Model evaluation utilities
└── main.py                      # Main execution script
```

## Installation

1. Clone this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Download the dataset from Kaggle and place it in the project directory
2. Run the main script:
```bash
python main.py
```

## Model Architecture

The CNN model consists of:
- 3 Convolutional blocks (Conv2D + MaxPooling)
- Flatten layer
- 3 Dense layers with BatchNormalization and Dropout
- Softmax output layer for 3-class classification

## Training Configuration

- **Image Size**: 256x256
- **Batch Size**: 64
- **Epochs**: 10 (with early stopping)
- **Train/Val Split**: 80/20
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

## Features

- Modular and maintainable code structure
- Configurable hyperparameters
- Custom callbacks for training control
- Comprehensive visualization utilities
- Detailed model evaluation metrics

## Results

The model achieves high accuracy on normal lung tissue detection, with room for improvement on cancerous tissue classification.

## References

- Original tutorial: [GeeksforGeeks - Lung Cancer Detection using CNN](https://www.geeksforgeeks.org/deep-learning/lung-cancer-detection-using-convolutional-neural-network-cnn/)
