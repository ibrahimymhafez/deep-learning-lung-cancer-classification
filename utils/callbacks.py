"""
Custom callbacks module.
Defines custom callbacks for training control.
"""

import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from config import config


class CustomAccuracyCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to stop training when validation accuracy reaches target.
    """
    
    def __init__(self, target_accuracy=None):
        """
        Initialize the callback.
        
        Args:
            target_accuracy (float): Target validation accuracy threshold
        """
        super().__init__()
        self.target_accuracy = target_accuracy or config.TARGET_ACCURACY
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch.
        
        Args:
            epoch (int): Current epoch number
            logs (dict): Dictionary containing training metrics
        """
        if logs is None:
            logs = {}
            
        val_accuracy = logs.get('val_accuracy', 0)
        
        if val_accuracy > self.target_accuracy:
            print(f'\n\nValidation accuracy has reached {val_accuracy:.2%}, '
                  f'exceeding target of {self.target_accuracy:.2%}.')
            print('Stopping further training.')
            self.model.stop_training = True


def get_callbacks(target_accuracy=None, early_stopping_patience=None, 
                  reduce_lr_patience=None, reduce_lr_factor=None):
    """
    Create and return a list of callbacks for model training.
    
    Args:
        target_accuracy (float): Target validation accuracy for custom callback
        early_stopping_patience (int): Patience for early stopping
        reduce_lr_patience (int): Patience for learning rate reduction
        reduce_lr_factor (float): Factor to reduce learning rate
        
    Returns:
        list: List of callback objects
    """
    # Use config values if not provided
    target_accuracy = target_accuracy or config.TARGET_ACCURACY
    early_stopping_patience = early_stopping_patience or config.EARLY_STOPPING_PATIENCE
    reduce_lr_patience = reduce_lr_patience or config.REDUCE_LR_PATIENCE
    reduce_lr_factor = reduce_lr_factor or config.REDUCE_LR_FACTOR
    
    # Early Stopping callback
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        monitor='val_accuracy',
        restore_best_weights=True,
        verbose=1,
        mode='max'
    )
    
    # Learning Rate Reduction callback
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        patience=reduce_lr_patience,
        factor=reduce_lr_factor,
        verbose=1,
        mode='min',
        min_lr=1e-7
    )
    
    # Custom accuracy callback
    custom_callback = CustomAccuracyCallback(target_accuracy=target_accuracy)
    
    callbacks = [early_stopping, reduce_lr, custom_callback]
    
    print("Callbacks configured:")
    print(f"  - Early Stopping (patience={early_stopping_patience})")
    print(f"  - Reduce LR on Plateau (patience={reduce_lr_patience}, factor={reduce_lr_factor})")
    print(f"  - Custom Accuracy Stop (target={target_accuracy:.2%})")
    
    return callbacks


if __name__ == "__main__":
    # Test callback creation
    callbacks = get_callbacks()
    print(f"\nCreated {len(callbacks)} callbacks successfully!")
