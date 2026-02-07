# Quick Start Guide

## Deep Learning-Based Multi-Class Lung Cancer Classification

This guide will help you get started with the Lung Cancer Classification project quickly.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Setup Instructions

### 1. Install Dependencies

```bash
cd "/Users/ibrahimyaser/Documents/CS Study/My Projects/Lung Cancer Detection"
pip install -r requirements.txt
```

### 2. Download Dataset

Download the dataset from Kaggle:
- URL: https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images
- Place the downloaded `lung-and-colon-cancer-histopathological-images.zip` file in the project root directory

### 3. Run the Project

```bash
python main.py
```

This will:
1. Extract the dataset
2. Visualize sample images
3. Preprocess the data
4. Build the CNN model
5. Train the model
6. Evaluate performance
7. Save the trained model

## Project Structure Overview

```
Lung Cancer Detection/
├── main.py                      # Main execution script (START HERE)
├── requirements.txt             # Python dependencies
├── README.md                    # Full documentation
├── QUICKSTART.md               # This file
│
├── config/                      # Configuration settings
│   ├── __init__.py
│   └── config.py               # All hyperparameters
│
├── data/                        # Data loading
│   ├── __init__.py
│   └── data_loader.py          # Dataset extraction & management
│
├── preprocessing/               # Data preprocessing
│   ├── __init__.py
│   └── image_processor.py      # Image loading & preparation
│
├── models/                      # Model architecture
│   ├── __init__.py
│   └── cnn_model.py            # CNN model definition
│
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── callbacks.py            # Training callbacks
│   └── visualization.py        # Visualization utilities
│
├── training/                    # Training logic
│   ├── __init__.py
│   └── trainer.py              # Model training
│
└── evaluation/                  # Model evaluation
    ├── __init__.py
    └── evaluator.py            # Performance metrics
```

## Module Usage Examples

### Using Individual Modules

#### 1. Data Loading
```python
from data.data_loader import DataLoader

loader = DataLoader()
loader.extract_dataset()
loader.verify_dataset()
```

#### 2. Image Processing
```python
from preprocessing.image_processor import ImageProcessor

processor = ImageProcessor()
X_train, X_val, Y_train, Y_val = processor.prepare_full_dataset()
```

#### 3. Model Building
```python
from models.cnn_model import create_model

cnn = create_model()
cnn.get_model_summary()
```

#### 4. Training
```python
from training.trainer import ModelTrainer

trainer = ModelTrainer(cnn.model)
history = trainer.train(X_train, Y_train, X_val, Y_val)
```

#### 5. Evaluation
```python
from evaluation.evaluator import ModelEvaluator

evaluator = ModelEvaluator(cnn.model)
results = evaluator.full_evaluation(X_val, Y_val)
```

#### 6. Visualization
```python
from utils.visualization import Visualizer

viz = Visualizer()
viz.visualize_sample_images()
viz.plot_training_history(history)
```

## Configuration

All hyperparameters can be modified in `config/config.py`:

- **Image Size**: `IMG_SIZE = 256`
- **Batch Size**: `BATCH_SIZE = 64`
- **Epochs**: `EPOCHS = 10`
- **Learning Rate**: `LEARNING_RATE = 0.001`
- **Train/Val Split**: `SPLIT_RATIO = 0.2`

## Expected Output

After running `main.py`, you should see:
1. Dataset extraction confirmation
2. Sample images from each class
3. Dataset statistics (15,000 images total)
4. Model architecture summary
5. Training progress with accuracy/loss metrics
6. Training history plots
7. Classification report with per-class metrics
8. Confusion matrix
9. Saved model location

## Troubleshooting

### Issue: Dataset not found
**Solution**: Download the dataset from Kaggle and place it in the project root

### Issue: Out of memory
**Solution**: Reduce `BATCH_SIZE` in `config/config.py`

### Issue: Import errors
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

### Issue: Slow training
**Solution**: Consider using GPU acceleration or reduce `IMG_SIZE` in config

## Next Steps

1. Experiment with different hyperparameters in `config/config.py`
2. Try different model architectures in `models/cnn_model.py`
3. Add data augmentation in `preprocessing/image_processor.py`
4. Implement cross-validation for better evaluation
5. Deploy the model for inference

## Support

For issues or questions, refer to:
- Main README.md for detailed documentation
- ARCHITECTURE.md for system design details
- Dataset source: [Kaggle LC25000 Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
