import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.metrics import roc_curve, auc

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image"""
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    return img_array

def analyze_image(model, image_path):
    """Analyze a single image and return prediction score"""
    img_array = load_and_preprocess_image(image_path)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)[0][0]
    return prediction

def plot_threshold_analysis(y_true, y_pred):
    """Plot ROC curve and find optimal threshold"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (closest to top-left corner)
    optimal_idx = np.argmin(np.sqrt((1-tpr)**2 + fpr**2))
    optimal_threshold = thresholds[optimal_idx]
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', marker='o', label=f'Optimal threshold = {optimal_threshold:.2f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Optimal Threshold')
    plt.legend(loc="lower right")
    plt.savefig('static/roc_curve.png')
    plt.close()
    
    return optimal_threshold

def main():
    # Load the model
    print("Loading model...")
    model = tf.keras.models.load_model('van_gogh_detector.keras')
    print("Model loaded successfully!")
    
    # Analyze authentic images
    print("\nAnalyzing authentic images:")
    authentic_dir = 'data/test/authentic'
    authentic_scores = []
    authentic_files = []
    
    for filename in os.listdir(authentic_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(authentic_dir, filename)
            score = analyze_image(model, img_path)
            authentic_scores.append(score)
            authentic_files.append(filename)
            print(f"{filename}: {score:.4f} ({'Forged' if score > 0.5 else 'Authentic'})")
    
    # Analyze forged images
    print("\nAnalyzing forged images:")
    forged_dir = 'data/test/forged'
    forged_scores = []
    forged_files = []
    
    for filename in os.listdir(forged_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(forged_dir, filename)
            score = analyze_image(model, img_path)
            forged_scores.append(score)
            forged_files.append(filename)
            print(f"{filename}: {score:.4f} ({'Forged' if score > 0.5 else 'Authentic'})")
    
    # Combine all scores and true labels
    all_scores = np.array(authentic_scores + forged_scores)
    all_labels = np.array([0] * len(authentic_scores) + [1] * len(forged_scores))
    
    # Find optimal threshold
    optimal_threshold = plot_threshold_analysis(all_labels, all_scores)
    
    # Plot score distributions
    plt.figure(figsize=(12, 6))
    plt.hist(authentic_scores, bins=20, alpha=0.5, label='Authentic', color='green')
    plt.hist(forged_scores, bins=20, alpha=0.5, label='Forged', color='red')
    plt.axvline(x=optimal_threshold, color='black', linestyle='--', label=f'Optimal Threshold = {optimal_threshold:.2f}')
    plt.axvline(x=0.5, color='gray', linestyle='--', label='Current Threshold = 0.5')
    plt.xlabel('Prediction Score')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Prediction Scores')
    plt.legend()
    plt.savefig('static/score_distribution.png')
    plt.close()
    
    # Save detailed analysis
    with open('static/prediction_analysis.txt', 'w') as f:
        f.write("Prediction Analysis Report\n")
        f.write("========================\n\n")
        
        f.write("Authentic Images:\n")
        f.write("-----------------\n")
        for filename, score in zip(authentic_files, authentic_scores):
            f.write(f"{filename}: {score:.4f} ({'Forged' if score > 0.5 else 'Authentic'})\n")
        
        f.write("\nForged Images:\n")
        f.write("-------------\n")
        for filename, score in zip(forged_files, forged_scores):
            f.write(f"{filename}: {score:.4f} ({'Forged' if score > 0.5 else 'Authentic'})\n")
        
        f.write(f"\nOptimal threshold: {optimal_threshold:.4f}\n")
        
        # Calculate metrics with optimal threshold
        y_pred_optimal = (all_scores > optimal_threshold).astype(int)
        accuracy_optimal = np.mean(y_pred_optimal == all_labels)
        f.write(f"Accuracy with optimal threshold: {accuracy_optimal:.4f}\n")

if __name__ == '__main__':
    main() 