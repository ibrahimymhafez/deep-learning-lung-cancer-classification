"""
Model evaluation module.
Handles model evaluation and performance metrics.
"""

import numpy as np
from sklearn import metrics
from config import config


class ModelEvaluator:
    """
    Handles model evaluation and performance metrics calculation.
    """
    
    def __init__(self, model, class_names=None):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained Keras model
            class_names (list): List of class names
        """
        self.model = model
        self.class_names = class_names or config.CLASSES
        
    def predict(self, X):
        """
        Generate predictions for input data.
        
        Args:
            X (numpy.ndarray): Input images
            
        Returns:
            numpy.ndarray: Predicted class probabilities
        """
        print(f"Generating predictions for {len(X)} samples...")
        predictions = self.model.predict(X)
        return predictions
    
    def evaluate(self, X_val, Y_val):
        """
        Evaluate model performance on validation data.
        
        Args:
            X_val (numpy.ndarray): Validation images
            Y_val (numpy.ndarray): Validation labels (one-hot encoded)
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        # Generate predictions
        Y_pred_probs = self.predict(X_val)
        
        # Convert one-hot encoded labels to class indices
        Y_true = np.argmax(Y_val, axis=1)
        Y_pred = np.argmax(Y_pred_probs, axis=1)
        
        # Calculate metrics
        accuracy = metrics.accuracy_score(Y_true, Y_pred)
        
        # Generate classification report
        report = metrics.classification_report(
            Y_true, Y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Generate confusion matrix
        conf_matrix = metrics.confusion_matrix(Y_true, Y_pred)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'y_true': Y_true,
            'y_pred': Y_pred,
            'y_pred_probs': Y_pred_probs
        }
        
        return results
    
    def print_evaluation_results(self, results):
        """
        Print formatted evaluation results.
        
        Args:
            results (dict): Dictionary containing evaluation metrics
        """
        print(f"\nOverall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print("\n" + "-"*70)
        print("CLASSIFICATION REPORT")
        print("-"*70)
        
        # Print classification report
        report_str = metrics.classification_report(
            results['y_true'], 
            results['y_pred'],
            target_names=self.class_names
        )
        print(report_str)
        
        print("-"*70)
        print("CONFUSION MATRIX")
        print("-"*70)
        print(results['confusion_matrix'])
        print()
        
        # Print per-class accuracy
        print("-"*70)
        print("PER-CLASS METRICS")
        print("-"*70)
        
        for class_name in self.class_names:
            class_metrics = results['classification_report'][class_name]
            print(f"\n{class_name}:")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall:    {class_metrics['recall']:.4f}")
            print(f"  F1-Score:  {class_metrics['f1-score']:.4f}")
            print(f"  Support:   {int(class_metrics['support'])}")
        
        print("\n" + "="*70 + "\n")
    
    def get_misclassified_samples(self, X_val, results, max_samples=10):
        """
        Get misclassified samples for analysis.
        
        Args:
            X_val (numpy.ndarray): Validation images
            results (dict): Evaluation results dictionary
            max_samples (int): Maximum number of samples to return
            
        Returns:
            dict: Dictionary containing misclassified sample information
        """
        Y_true = results['y_true']
        Y_pred = results['y_pred']
        
        # Find misclassified indices
        misclassified_idx = np.where(Y_true != Y_pred)[0]
        
        if len(misclassified_idx) == 0:
            print("No misclassified samples found!")
            return None
        
        # Limit to max_samples
        if len(misclassified_idx) > max_samples:
            misclassified_idx = misclassified_idx[:max_samples]
        
        misclassified_data = {
            'indices': misclassified_idx,
            'images': X_val[misclassified_idx],
            'true_labels': Y_true[misclassified_idx],
            'predicted_labels': Y_pred[misclassified_idx],
            'predicted_probs': results['y_pred_probs'][misclassified_idx]
        }
        
        print(f"\nFound {len(misclassified_idx)} misclassified samples (showing up to {max_samples})")
        
        return misclassified_data
    
    def full_evaluation(self, X_val, Y_val):
        """
        Perform complete evaluation and print results.
        
        Args:
            X_val (numpy.ndarray): Validation images
            Y_val (numpy.ndarray): Validation labels
            
        Returns:
            dict: Evaluation results
        """
        results = self.evaluate(X_val, Y_val)
        self.print_evaluation_results(results)
        return results


if __name__ == "__main__":
    print("ModelEvaluator module loaded successfully!")
    print("Use this module to evaluate your trained model.")
