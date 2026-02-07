"""
Visualization utilities module.
Provides functions for visualizing data and training results.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
from config import config


class Visualizer:
    """
    Handles visualization of images and training results.
    """
    
    def __init__(self):
        """
        Initialize the Visualizer.
        """
        self.classes = config.CLASSES
        self.data_path = config.EXTRACTED_DATA_PATH
        
    def visualize_sample_images(self, samples_per_class=None):
        """
        Visualize random sample images from each class.
        
        Args:
            samples_per_class (int): Number of sample images to display per class
        """
        samples_per_class = samples_per_class or config.SAMPLE_IMAGES_PER_CLASS
        
        for class_name in self.classes:
            image_dir = f'{self.data_path}/{class_name}'
            
            if not os.path.exists(image_dir):
                print(f"Warning: Directory not found: {image_dir}")
                continue
            
            images = os.listdir(image_dir)
            
            if len(images) == 0:
                print(f"Warning: No images found in {image_dir}")
                continue
            
            # Create subplots
            fig, ax = plt.subplots(1, samples_per_class, figsize=config.FIGURE_SIZE)
            fig.suptitle(f'Sample images for {class_name} category', fontsize=16, fontweight='bold')
            
            # Display random samples
            for i in range(samples_per_class):
                k = np.random.randint(0, len(images))
                img_path = f'{image_dir}/{images[k]}'
                
                try:
                    img = np.array(Image.open(img_path))
                    ax[i].imshow(img)
                    ax[i].axis('off')
                    ax[i].set_title(f'Sample {i+1}', fontsize=10)
                except Exception as e:
                    print(f"Error loading image {img_path}: {str(e)}")
            
            plt.tight_layout()
            plt.show()
    
    def plot_training_history(self, history):
        """
        Plot training and validation accuracy/loss over epochs.
        
        Args:
            history: Training history object from model.fit()
        """
        # Convert history to DataFrame
        history_df = pd.DataFrame(history.history)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history_df['accuracy'], label='Training Accuracy', marker='o')
        ax1.plot(history_df['val_accuracy'], label='Validation Accuracy', marker='s')
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(history_df['loss'], label='Training Loss', marker='o')
        ax2.plot(history_df['val_loss'], label='Validation Loss', marker='s')
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_accuracy_only(self, history):
        """
        Plot only training and validation accuracy.
        
        Args:
            history: Training history object from model.fit()
        """
        history_df = pd.DataFrame(history.history)
        
        plt.figure(figsize=(10, 6))
        plt.plot(history_df['accuracy'], label='Training Accuracy', marker='o', linewidth=2)
        plt.plot(history_df['val_accuracy'], label='Validation Accuracy', marker='s', linewidth=2)
        plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def visualize_predictions(self, images, true_labels, predicted_labels, num_samples=9):
        """
        Visualize sample predictions with true and predicted labels.
        
        Args:
            images (numpy.ndarray): Array of images
            true_labels (numpy.ndarray): True class labels
            predicted_labels (numpy.ndarray): Predicted class labels
            num_samples (int): Number of samples to display
        """
        # Calculate grid size
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()
        
        for i in range(num_samples):
            if i >= len(images):
                break
            
            # Get random index
            idx = np.random.randint(0, len(images))
            
            # Display image
            axes[i].imshow(images[idx])
            axes[i].axis('off')
            
            # Set title with true and predicted labels
            true_class = self.classes[true_labels[idx]]
            pred_class = self.classes[predicted_labels[idx]]
            
            color = 'green' if true_labels[idx] == predicted_labels[idx] else 'red'
            axes[i].set_title(f'True: {true_class}\nPred: {pred_class}', 
                            color=color, fontsize=10)
        
        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Test the Visualizer
    viz = Visualizer()
    print("Visualizer initialized successfully!")
    print("Note: Run visualize_sample_images() after extracting the dataset.")
