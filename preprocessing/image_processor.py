"""
Image preprocessing module.
Handles image loading, resizing, and dataset preparation.
"""

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from config import config
from data.data_loader import DataLoader


class ImageProcessor:
    """
    Handles image preprocessing and dataset preparation.
    """
    
    def __init__(self, img_size=None):
        """
        Initialize ImageProcessor.
        
        Args:
            img_size (int): Target size for image resizing
        """
        self.img_size = img_size or config.IMG_SIZE
        self.data_loader = DataLoader()
        self.classes = config.CLASSES
        
    def load_and_resize_image(self, image_path):
        """
        Load and resize a single image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Resized image array
        """
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        return img_resized
    
    def prepare_dataset(self):
        """
        Load all images, resize them, and prepare labels.
        
        Returns:
            tuple: (X, Y) where X is image array and Y is label array
        """
        print("Loading and preprocessing images...")
        X = []
        Y = []
        
        for i, class_name in enumerate(self.classes):
            print(f"Processing class '{class_name}' (label {i})...")
            image_paths = self.data_loader.get_image_paths(class_name)
            
            for image_path in image_paths:
                try:
                    img = self.load_and_resize_image(image_path)
                    X.append(img)
                    Y.append(i)
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    continue
        
        # Convert to numpy arrays
        X = np.asarray(X)
        print(f"\nTotal images loaded: {len(X)}")
        print(f"Image shape: {X.shape}")
        
        return X, Y
    
    def encode_labels(self, Y):
        """
        One-hot encode the labels.
        
        Args:
            Y (list): List of class labels
            
        Returns:
            numpy.ndarray: One-hot encoded labels
        """
        one_hot_encoded_Y = pd.get_dummies(Y).values
        print(f"Label encoding completed. Shape: {one_hot_encoded_Y.shape}")
        return one_hot_encoded_Y
    
    def split_dataset(self, X, Y, test_size=None, random_state=None):
        """
        Split dataset into training and validation sets.
        
        Args:
            X (numpy.ndarray): Image data
            Y (numpy.ndarray): Labels
            test_size (float): Proportion of validation data
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_val, Y_train, Y_val)
        """
        test_size = test_size or config.SPLIT_RATIO
        random_state = random_state or config.RANDOM_STATE
        
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        print(f"\nDataset split completed:")
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        
        return X_train, X_val, Y_train, Y_val
    
    def prepare_full_dataset(self):
        """
        Complete pipeline: load, preprocess, encode, and split dataset.
        
        Returns:
            tuple: (X_train, X_val, Y_train, Y_val)
        """
        # Load and preprocess images
        X, Y = self.prepare_dataset()
        
        # Encode labels
        Y_encoded = self.encode_labels(Y)
        
        # Split dataset
        X_train, X_val, Y_train, Y_val = self.split_dataset(X, Y_encoded)
        
        return X_train, X_val, Y_train, Y_val


if __name__ == "__main__":
    # Test the ImageProcessor
    processor = ImageProcessor()
    X_train, X_val, Y_train, Y_val = processor.prepare_full_dataset()
    print("\nDataset preparation completed successfully!")
