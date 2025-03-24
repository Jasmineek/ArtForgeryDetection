import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def load_and_prepare_data():
    """Load and prepare test data"""
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )
    
    return test_generator

def generate_confusion_matrix():
    """Generate and display confusion matrix"""
    # Load the model
    model = tf.keras.models.load_model('van_gogh_detector_final.keras')
    
    # Load test data
    test_generator = load_and_prepare_data()
    
    # Get predictions
    predictions = model.predict(test_generator)
    y_pred = (predictions > 0.5).astype(int)
    y_true = test_generator.classes
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Authentic', 'Forged'],
                yticklabels=['Authentic', 'Forged'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                              target_names=['Authentic', 'Forged']))
    
    # Calculate and print additional metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    print("\nDetailed Metrics:")
    print(f"True Negatives (Correctly identified authentic): {tn}")
    print(f"False Positives (Incorrectly identified as authentic): {fp}")
    print(f"False Negatives (Missed forgeries): {fn}")
    print(f"True Positives (Correctly identified forgeries): {tp}")
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == '__main__':
    generate_confusion_matrix() 