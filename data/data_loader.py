"""
Data loading and extraction module.
Handles dataset extraction and file path management.
"""

import os
from zipfile import ZipFile
from glob import glob
from config import config


class DataLoader:
    """
    Handles dataset extraction and file path management.
    """
    
    def __init__(self, dataset_path=None, extracted_path=None):
        """
        Initialize DataLoader with dataset paths.
        
        Args:
            dataset_path (str): Path to the zip file containing the dataset
            extracted_path (str): Path where data will be extracted
        """
        self.dataset_path = dataset_path or config.DATASET_PATH
        self.extracted_path = extracted_path or config.EXTRACTED_DATA_PATH
        self.classes = config.CLASSES
        
    def extract_dataset(self):
        """
        Extract the dataset from zip file.
        
        Returns:
            bool: True if extraction successful, False otherwise
        """
        try:
            if not os.path.exists(self.dataset_path):
                print(f"Error: Dataset not found at {self.dataset_path}")
                print("Please download the dataset from Kaggle:")
                print("https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images")
                return False
                
            print(f"Extracting dataset from {self.dataset_path}...")
            with ZipFile(self.dataset_path, 'r') as zip_ref:
                zip_ref.extractall()
            print("Dataset extraction completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error during extraction: {str(e)}")
            return False
    
    def get_image_paths(self, class_name):
        """
        Get all image paths for a specific class.
        
        Args:
            class_name (str): Name of the class (e.g., 'lung_n', 'lung_aca', 'lung_scc')
            
        Returns:
            list: List of image file paths
        """
        pattern = f'{self.extracted_path}/{class_name}/*.jpeg'
        return glob(pattern)
    
    def get_all_image_paths(self):
        """
        Get all image paths organized by class.
        
        Returns:
            dict: Dictionary with class names as keys and list of image paths as values
        """
        image_paths = {}
        for class_name in self.classes:
            image_paths[class_name] = self.get_image_paths(class_name)
        return image_paths
    
    def verify_dataset(self):
        """
        Verify that the dataset has been extracted and contains expected classes.
        
        Returns:
            bool: True if dataset is valid, False otherwise
        """
        if not os.path.exists(self.extracted_path):
            print(f"Error: Extracted data path not found: {self.extracted_path}")
            return False
        
        for class_name in self.classes:
            class_path = f'{self.extracted_path}/{class_name}'
            if not os.path.exists(class_path):
                print(f"Error: Class directory not found: {class_path}")
                return False
            
            images = self.get_image_paths(class_name)
            print(f"Found {len(images)} images for class '{class_name}'")
        
        return True


if __name__ == "__main__":
    # Test the DataLoader
    loader = DataLoader()
    
    # Extract dataset
    if loader.extract_dataset():
        # Verify dataset
        if loader.verify_dataset():
            print("\nDataset is ready for use!")
        else:
            print("\nDataset verification failed!")
