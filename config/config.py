"""
Configuration file for Lung Cancer Detection project.
Contains all hyperparameters and settings.
"""

import os

# Dataset Configuration
DATASET_PATH = 'lung-and-colon-cancer-histopathological-images.zip'
EXTRACTED_DATA_PATH = 'lung_colon_image_set/lung_image_sets'
CLASSES = ['lung_aca', 'lung_n', 'lung_scc']

# Image Processing Configuration
IMG_SIZE = 256
IMAGE_CHANNELS = 3

# Training Configuration
SPLIT_RATIO = 0.2  # 80% train, 20% validation
EPOCHS = 10
BATCH_SIZE = 64
RANDOM_STATE = 2022

# Model Configuration
CONV_FILTERS = [32, 64, 128]
KERNEL_SIZES = [(5, 5), (3, 3), (3, 3)]
POOL_SIZE = (2, 2)
DENSE_UNITS = [256, 128]
DROPOUT_RATE = 0.3
NUM_CLASSES = len(CLASSES)

# Optimizer Configuration
LEARNING_RATE = 0.001
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'categorical_crossentropy'
METRICS = ['accuracy']

# Callback Configuration
EARLY_STOPPING_PATIENCE = 3
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5
TARGET_ACCURACY = 0.90

# Visualization Configuration
SAMPLE_IMAGES_PER_CLASS = 3
FIGURE_SIZE = (15, 5)

# Model Saving Configuration
MODEL_SAVE_PATH = 'saved_models'
MODEL_NAME = 'lung_cancer_cnn_model.h5'

# Create necessary directories
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
