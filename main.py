"""
Deep Learning-Based Multi-Class Lung Cancer Classification

This script orchestrates the entire pipeline from data loading to model evaluation
for classifying lung histopathological images into normal, adenocarcinoma, and 
squamous cell carcinoma categories.
"""

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from data.data_loader import DataLoader
from preprocessing.image_processor import ImageProcessor
from models.cnn_model import create_model
from training.trainer import ModelTrainer
from evaluation.evaluator import ModelEvaluator
from utils.visualization import Visualizer
from config import config


def main():
    """
    Main execution function.
    """
    print("\n" + "="*70)
    print("DEEP LEARNING-BASED MULTI-CLASS LUNG CANCER CLASSIFICATION")
    print("="*70 + "\n")
    
    # Step 1: Load and extract dataset
    print("Step 1: Loading Dataset")
    print("-" * 70)
    data_loader = DataLoader()
    
    if not os.path.exists(config.EXTRACTED_DATA_PATH):
        if not data_loader.extract_dataset():
            print("Failed to extract dataset. Please ensure the dataset file is present.")
            return
    else:
        print("Dataset already extracted.")
    
    # Verify dataset
    if not data_loader.verify_dataset():
        print("Dataset verification failed. Please check the dataset.")
        return
    
    print()
    
    # Step 2: Visualize sample images (optional)
    print("Step 2: Visualizing Sample Images")
    print("-" * 70)
    visualizer = Visualizer()
    try:
        visualizer.visualize_sample_images()
    except Exception as e:
        print(f"Visualization skipped: {str(e)}")
    
    print()
    
    # Step 3: Prepare dataset
    print("Step 3: Preparing Dataset")
    print("-" * 70)
    processor = ImageProcessor()
    X_train, X_val, Y_train, Y_val = processor.prepare_full_dataset()
    print()
    
    # Step 4: Build and compile model
    print("Step 4: Building CNN Model")
    print("-" * 70)
    cnn = create_model()
    print("\nModel Architecture:")
    cnn.get_model_summary()
    print()
    
    # Step 5: Train model
    print("Step 5: Training Model")
    print("-" * 70)
    trainer = ModelTrainer(cnn.model)
    history = trainer.train(X_train, Y_train, X_val, Y_val)
    trainer.print_training_summary()
    
    # Step 6: Visualize training history
    print("Step 6: Visualizing Training History")
    print("-" * 70)
    try:
        visualizer.plot_training_history(history)
    except Exception as e:
        print(f"Visualization error: {str(e)}")
    
    print()
    
    # Step 7: Evaluate model
    print("Step 7: Evaluating Model")
    print("-" * 70)
    evaluator = ModelEvaluator(cnn.model)
    results = evaluator.full_evaluation(X_val, Y_val)
    
    # Step 8: Save model
    print("Step 8: Saving Model")
    print("-" * 70)
    try:
        cnn.save_model()
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
    
    print()
    
    # Final summary
    print("="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nFinal Validation Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}/{config.MODEL_NAME}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
