"""
CNN Model architecture module.
Defines the Convolutional Neural Network for lung cancer detection.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
from config import config


class LungCancerCNN:
    """
    CNN model for lung cancer detection.
    """
    
    def __init__(self):
        """
        Initialize the CNN model builder.
        """
        self.img_size = config.IMG_SIZE
        self.num_classes = config.NUM_CLASSES
        self.model = None
        
    def build_model(self):
        """
        Build the CNN model architecture.
        
        Returns:
            keras.Model: Compiled CNN model
        """
        model = keras.models.Sequential([
            # First Convolutional Block
            layers.Conv2D(
                filters=config.CONV_FILTERS[0],
                kernel_size=config.KERNEL_SIZES[0],
                activation='relu',
                input_shape=(self.img_size, self.img_size, config.IMAGE_CHANNELS),
                padding='same',
                name='conv_block_1'
            ),
            layers.MaxPooling2D(config.POOL_SIZE, name='maxpool_1'),
            
            # Second Convolutional Block
            layers.Conv2D(
                filters=config.CONV_FILTERS[1],
                kernel_size=config.KERNEL_SIZES[1],
                activation='relu',
                padding='same',
                name='conv_block_2'
            ),
            layers.MaxPooling2D(config.POOL_SIZE, name='maxpool_2'),
            
            # Third Convolutional Block
            layers.Conv2D(
                filters=config.CONV_FILTERS[2],
                kernel_size=config.KERNEL_SIZES[2],
                activation='relu',
                padding='same',
                name='conv_block_3'
            ),
            layers.MaxPooling2D(config.POOL_SIZE, name='maxpool_3'),
            
            # Flatten layer
            layers.Flatten(name='flatten'),
            
            # First Dense Block
            layers.Dense(config.DENSE_UNITS[0], activation='relu', name='dense_1'),
            layers.BatchNormalization(name='batch_norm_1'),
            
            # Second Dense Block
            layers.Dense(config.DENSE_UNITS[1], activation='relu', name='dense_2'),
            layers.Dropout(config.DROPOUT_RATE, name='dropout'),
            layers.BatchNormalization(name='batch_norm_2'),
            
            # Output Layer
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, optimizer=None, loss=None, metrics=None):
        """
        Compile the model with optimizer, loss function, and metrics.
        
        Args:
            optimizer (str): Optimizer name
            loss (str): Loss function name
            metrics (list): List of metrics to track
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation. Call build_model() first.")
        
        optimizer = optimizer or config.OPTIMIZER
        loss = loss or config.LOSS_FUNCTION
        metrics = metrics or config.METRICS
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        print("Model compiled successfully!")
    
    def get_model_summary(self):
        """
        Print model architecture summary.
        """
        if self.model is None:
            raise ValueError("Model must be built first. Call build_model().")
        
        return self.model.summary()
    
    def save_model(self, filepath=None):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Build and train the model first.")
        
        filepath = filepath or f"{config.MODEL_SAVE_PATH}/{config.MODEL_NAME}"
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """
        Load a saved model.
        
        Args:
            filepath (str): Path to the saved model
        """
        filepath = filepath or f"{config.MODEL_SAVE_PATH}/{config.MODEL_NAME}"
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model


def create_model():
    """
    Convenience function to create and compile the model.
    
    Returns:
        keras.Model: Compiled CNN model
    """
    cnn = LungCancerCNN()
    model = cnn.build_model()
    cnn.compile_model()
    return cnn


if __name__ == "__main__":
    # Test the model creation
    cnn = create_model()
    print("\nModel Architecture:")
    cnn.get_model_summary()
