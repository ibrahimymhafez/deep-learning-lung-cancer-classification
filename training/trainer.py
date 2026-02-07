"""
Model training module.
Handles the training process of the CNN model.
"""

import warnings
warnings.filterwarnings('ignore')

from config import config
from utils.callbacks import get_callbacks


class ModelTrainer:
    """
    Handles model training with configured callbacks.
    """
    
    def __init__(self, model):
        """
        Initialize the trainer with a model.
        
        Args:
            model: Compiled Keras model
        """
        self.model = model
        self.history = None
        
    def train(self, X_train, Y_train, X_val, Y_val, 
              epochs=None, batch_size=None, callbacks=None, verbose=1):
        """
        Train the model on the provided data.
        
        Args:
            X_train (numpy.ndarray): Training images
            Y_train (numpy.ndarray): Training labels
            X_val (numpy.ndarray): Validation images
            Y_val (numpy.ndarray): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            callbacks (list): List of callbacks
            verbose (int): Verbosity mode (0, 1, or 2)
            
        Returns:
            History object containing training metrics
        """
        epochs = epochs or config.EPOCHS
        batch_size = batch_size or config.BATCH_SIZE
        
        if callbacks is None:
            callbacks = get_callbacks()
        
        print("\n" + "="*70)
        print("STARTING MODEL TRAINING")
        print("="*70)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print("="*70 + "\n")
        
        self.history = self.model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        
        return self.history
    
    def get_training_summary(self):
        """
        Get a summary of training results.
        
        Returns:
            dict: Dictionary containing training metrics
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return None
        
        history_dict = self.history.history
        
        # Get final metrics
        final_train_acc = history_dict['accuracy'][-1]
        final_val_acc = history_dict['val_accuracy'][-1]
        final_train_loss = history_dict['loss'][-1]
        final_val_loss = history_dict['val_loss'][-1]
        
        # Get best metrics
        best_val_acc = max(history_dict['val_accuracy'])
        best_val_acc_epoch = history_dict['val_accuracy'].index(best_val_acc) + 1
        
        summary = {
            'total_epochs': len(history_dict['accuracy']),
            'final_train_accuracy': final_train_acc,
            'final_val_accuracy': final_val_acc,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'best_val_accuracy': best_val_acc,
            'best_val_accuracy_epoch': best_val_acc_epoch
        }
        
        return summary
    
    def print_training_summary(self):
        """
        Print a formatted summary of training results.
        """
        summary = self.get_training_summary()
        
        if summary is None:
            return
        
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        print(f"Total Epochs Trained: {summary['total_epochs']}")
        print(f"\nFinal Metrics:")
        print(f"  Training Accuracy:   {summary['final_train_accuracy']:.4f}")
        print(f"  Validation Accuracy: {summary['final_val_accuracy']:.4f}")
        print(f"  Training Loss:       {summary['final_train_loss']:.4f}")
        print(f"  Validation Loss:     {summary['final_val_loss']:.4f}")
        print(f"\nBest Validation Accuracy: {summary['best_val_accuracy']:.4f} "
              f"(Epoch {summary['best_val_accuracy_epoch']})")
        print("="*70 + "\n")


if __name__ == "__main__":
    print("ModelTrainer module loaded successfully!")
    print("Use this module to train your CNN model.")
