import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def plot_confusion_matrix(y_true, y_pred, threshold):
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure and axes
    plt.figure(figsize=(12, 8))
    
    # Create a subplot for the counts
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Authentic', 'Forged'],
                yticklabels=['Authentic', 'Forged'])
    plt.title(f'Confusion Matrix - Counts\n(threshold: {threshold:.2f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Create a subplot for the percentages
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdYlBu_r',
                xticklabels=['Authentic', 'Forged'],
                yticklabels=['Authentic', 'Forged'])
    plt.title('Confusion Matrix - Percentages')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add metrics text box
    tn, fp, fn, tp = cm.ravel()
    total = np.sum(cm)
    accuracy = (tp + tn) / total * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f'Metrics:\n\n' \
                  f'Accuracy: {accuracy:.1f}%\n' \
                  f'Precision: {precision:.1f}%\n' \
                  f'Recall: {recall:.1f}%\n' \
                  f'F1-Score: {f1:.1f}%\n\n' \
                  f'True Negatives (Authentic): {tn}\n' \
                  f'False Positives: {fp}\n' \
                  f'False Negatives: {fn}\n' \
                  f'True Positives (Forged): {tp}'
    
    plt.figtext(1.15, 0.5, metrics_text, fontsize=10, ha='left', va='center')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1.3, 1])
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Return metrics for potential threshold optimization
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }

def find_optimal_threshold(scores, y_true):
    """Find the optimal threshold by maximizing F1-score"""
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = None
    
    # Test different thresholds
    for threshold in np.arange(0.4, 0.8, 0.01):
        y_pred = (scores > threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': (tp + tn) / len(y_true)
            }
    
    return best_threshold, best_metrics

def test_directory(model, directory, expected_label, label_name, all_true_labels, all_predictions, all_scores):
    correct = 0
    total = 0
    scores = []
    
    print(f"\nTesting {label_name} images from {directory}:")
    print("-" * 80)
    
    for image_file in sorted(os.listdir(directory)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, image_file)
            img_array = load_and_preprocess_image(image_path)
            prediction = model.predict(img_array, verbose=0)[0][0]
            predicted_label = prediction > 0.62  # Using the same threshold as app.py
            is_correct = predicted_label == expected_label
            
            # Store for confusion matrix
            all_true_labels.append(1 if expected_label else 0)
            all_predictions.append(1 if predicted_label else 0)
            all_scores.append(prediction)
            
            scores.append(prediction)
            if is_correct:
                correct += 1
            total += 1
            
            status = "✓" if is_correct else "✗"
            print(f"{status} {image_file:40} Score: {prediction:.3f} ({'Forged' if predicted_label else 'Authentic'})")
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    avg_score = np.mean(scores) if scores else 0
    print(f"\nSummary for {label_name} images:")
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{total} correct)")
    print(f"Average score: {avg_score:.3f}")
    print(f"Score range: {min(scores):.3f} - {max(scores):.3f}")
    return accuracy, avg_score, min(scores), max(scores)

def main():
    print("Loading model...")
    model = tf.keras.models.load_model('van_gogh_detector.h5')
    model.summary()
    
    # Lists to store all predictions for confusion matrix
    all_true_labels = []
    all_predictions = []
    all_scores = []
    
    # Test authentic images
    authentic_acc, authentic_avg, authentic_min, authentic_max = test_directory(
        model, 'data/authentic', False, 'Authentic', all_true_labels, all_predictions, all_scores)
    
    # Test forged images
    forged_acc, forged_avg, forged_min, forged_max = test_directory(
        model, 'data/forged', True, 'Forged', all_true_labels, all_predictions, all_scores)
    
    # Find optimal threshold
    optimal_threshold, optimal_metrics = find_optimal_threshold(all_scores, all_true_labels)
    print(f"\nOptimal threshold: {optimal_threshold:.3f}")
    print(f"Optimal metrics:")
    print(f"  F1-Score: {optimal_metrics['f1']:.3f}")
    print(f"  Precision: {optimal_metrics['precision']:.3f}")
    print(f"  Recall: {optimal_metrics['recall']:.3f}")
    print(f"  Accuracy: {optimal_metrics['accuracy']:.3f}")
    
    # Plot confusion matrix with optimal threshold
    plot_confusion_matrix(all_true_labels, all_predictions, optimal_threshold)
    
    # Print classification report
    print("\nDetailed Classification Report:")
    print("-" * 80)
    print(classification_report(all_true_labels, all_predictions, 
                              target_names=['Authentic', 'Forged'],
                              digits=3))
    
    print("\nOverall Analysis:")
    print("-" * 80)
    print(f"Authentic images: {authentic_acc:.1f}% accuracy, scores {authentic_min:.3f} - {authentic_max:.3f} (avg: {authentic_avg:.3f})")
    print(f"Forged images: {forged_acc:.1f}% accuracy, scores {forged_min:.3f} - {forged_max:.3f} (avg: {forged_avg:.3f})")
    
    if authentic_max > optimal_threshold or forged_min < optimal_threshold:
        print("\nPotential Issues Detected:")
        if authentic_max > optimal_threshold:
            print(f"- Some authentic images are being classified as forged (max score: {authentic_max:.3f})")
        if forged_min < optimal_threshold:
            print(f"- Some forged images are being classified as authentic (min score: {forged_min:.3f})")
        print("\nRecommended fixes:")
        print("1. Retrain the model with more authentic data")
        print("2. Add more data augmentation for authentic images")
        print("3. Consider using class weights to balance the training")

if __name__ == '__main__':
    main() 