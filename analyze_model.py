import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
from scipy import stats

def load_and_prepare_data():
    """Load and prepare test data"""
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        classes=['authentic', 'forged'],
        shuffle=False
    )
    
    return test_generator

def analyze_model():
    """Analyze model performance"""
    # Load the model
    print("Loading model...")
    model = tf.keras.models.load_model('van_gogh_detector_final.keras')
    
    # Load test data
    print("Loading test data...")
    test_generator = load_and_prepare_data()
    
    # Get predictions
    print("Making predictions...")
    predictions = model.predict(test_generator)
    y_true = test_generator.classes
    
    # Calculate metrics at different thresholds
    thresholds = np.arange(0.1, 1.0, 0.1)
    best_f1 = 0
    best_threshold = 0.5
    metrics_by_threshold = {}
    
    for threshold in thresholds:
        y_pred = (predictions > threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_by_threshold[threshold] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {
                'tn': tn, 'fp': fp,
                'fn': fn, 'tp': tp
            }
        }
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Plot confidence distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=predictions[y_true==0], label='Authentic', alpha=0.5, bins=30)
    sns.histplot(data=predictions[y_true==1], label='Forged', alpha=0.5, bins=30)
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Distribution of Confidence Scores')
    plt.legend()
    plt.savefig('confidence_distribution.png')
    plt.close()
    
    # Save detailed analysis
    with open('detailed_analysis.txt', 'w') as f:
        f.write("Van Gogh Artwork Authentication Model Analysis\n")
        f.write("===========================================\n\n")
        
        f.write(f"Best Classification Threshold: {best_threshold:.2f}\n")
        f.write(f"ROC AUC Score: {roc_auc:.4f}\n\n")
        
        f.write("Performance at Different Thresholds:\n")
        f.write("===================================\n")
        
        for threshold, metrics in metrics_by_threshold.items():
            f.write(f"\nThreshold: {threshold:.1f}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1']:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(f"  True Negatives (Correct Authentic): {metrics['confusion_matrix']['tn']}\n")
            f.write(f"  False Positives (Wrong Authentic): {metrics['confusion_matrix']['fp']}\n")
            f.write(f"  False Negatives (Wrong Forged): {metrics['confusion_matrix']['fn']}\n")
            f.write(f"  True Positives (Correct Forged): {metrics['confusion_matrix']['tp']}\n")
    
    # Print summary
    print(f"\nAnalysis complete! Best threshold: {best_threshold:.2f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nDetailed results have been saved to 'detailed_analysis.txt'")
    print("ROC curve has been saved to 'roc_curve.png'")
    print("Confidence distribution has been saved to 'confidence_distribution.png'")
    
    return best_threshold, metrics_by_threshold[best_threshold]

if __name__ == '__main__':
    analyze_model() 