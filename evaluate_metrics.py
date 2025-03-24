import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import os

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image"""
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    return img_array

def load_test_data():
    """Load test data from the test directory"""
    X_test = []
    y_test = []
    
    # Load authentic images
    authentic_dir = 'data/test/authentic'
    for filename in os.listdir(authentic_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(authentic_dir, filename)
            try:
                img_array = load_and_preprocess_image(img_path)
                X_test.append(img_array)
                y_test.append(0)  # Authentic class
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
    
    # Load forged images
    forged_dir = 'data/test/forged'
    for filename in os.listdir(forged_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(forged_dir, filename)
            try:
                img_array = load_and_preprocess_image(img_path)
                X_test.append(img_array)
                y_test.append(1)  # Forged class
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
    
    return np.array(X_test), np.array(y_test)

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix with percentages"""
    plt.figure(figsize=(10, 8))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotation text with both count and percentage
    annotations = np.zeros_like(cm, dtype=str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)'
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title('Confusion Matrix\n(count and percentage)', pad=20)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    plt.savefig('static/confusion_matrix_detailed.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Load the model
    print("Loading model...")
    model = tf.keras.models.load_model('van_gogh_detector.keras')
    print("Model loaded successfully!")
    
    # Load test data
    print("\nLoading test data...")
    X_test, y_test = load_test_data()
    print(f"Loaded {len(X_test)} test images")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    class_names = ['Authentic', 'Forged']
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names)
    
    # Generate classification report
    report = classification_report(y_test, y_pred_classes, 
                                 target_names=class_names,
                                 digits=4)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Print detailed results
    print("\nDetailed Metrics:")
    print("================")
    print(f"\nConfusion Matrix:")
    print("----------------")
    print(f"True Negatives (Authentic correctly classified): {tn}")
    print(f"False Positives (Authentic incorrectly classified as Forged): {fp}")
    print(f"False Negatives (Forged incorrectly classified as Authentic): {fn}")
    print(f"True Positives (Forged correctly classified): {tp}")
    
    print("\nPerformance Metrics:")
    print("-------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    print("\nDetailed Classification Report:")
    print("-----------------------------")
    print(report)
    
    # Save metrics to file
    with open('static/detailed_metrics.txt', 'w') as f:
        f.write("Detailed Metrics Report\n")
        f.write("=====================\n\n")
        f.write("Confusion Matrix:\n")
        f.write("-----------------\n")
        f.write(f"True Negatives (Authentic correctly classified): {tn}\n")
        f.write(f"False Positives (Authentic incorrectly classified as Forged): {fp}\n")
        f.write(f"False Negatives (Forged incorrectly classified as Authentic): {fn}\n")
        f.write(f"True Positives (Forged correctly classified): {tp}\n\n")
        f.write("Performance Metrics:\n")
        f.write("-------------------\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1_score:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write("--------------------\n")
        f.write(report)

if __name__ == '__main__':
    main() 