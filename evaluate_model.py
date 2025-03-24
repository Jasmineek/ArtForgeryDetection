import tensorflow as tf
import numpy as np
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import datetime

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def load_test_data(test_dir):
    """Load test data from directory"""
    images = []
    labels = []
    
    # Load authentic images
    authentic_dir = os.path.join(test_dir, 'authentic')
    for img_name in os.listdir(authentic_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(authentic_dir, img_name)
            img_array = load_and_preprocess_image(img_path)
            images.append(img_array)
            labels.append(0)  # 0 for authentic
    
    # Load forged images
    forged_dir = os.path.join(test_dir, 'forged')
    for img_name in os.listdir(forged_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(forged_dir, img_name)
            img_array = load_and_preprocess_image(img_path)
            images.append(img_array)
            labels.append(1)  # 1 for forged
    
    return np.array(images), np.array(labels)

def plot_confusion_matrix(y_true, y_pred, save_path='static/confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, save_path='static/roc_curve.png'):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def create_evaluation_report(y_true, y_pred, y_pred_proba, save_path='static/evaluation_report.html'):
    """Create HTML evaluation report"""
    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Authentic', 'Forged'])
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    
    # Create HTML content
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .metric-box {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .metric-value {{ font-size: 1.2em; font-weight: bold; color: #3498db; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            pre {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <h2>Model Evaluation Report</h2>
        
        <div class="metric-box">
            <h3>Overall Accuracy</h3>
            <div class="metric-value">{accuracy:.4f}</div>
        </div>
        
        <div class="metric-box">
            <h3>Confusion Matrix</h3>
            <img src="confusion_matrix.png" alt="Confusion Matrix">
        </div>
        
        <div class="metric-box">
            <h3>ROC Curve</h3>
            <img src="roc_curve.png" alt="ROC Curve">
        </div>
        
        <div class="metric-box">
            <h3>Detailed Classification Report</h3>
            <pre>{report}</pre>
        </div>
    </body>
    </html>
    """
    
    with open(save_path, 'w') as f:
        f.write(html_content)

def evaluate_model():
    """Main evaluation function"""
    # Load the model
    model = tf.keras.models.load_model('van_gogh_detector.keras')
    print("Model loaded successfully!")
    
    # Load test data
    print("\nLoading test data...")
    X_test, y_test = load_test_data('test_data')
    print(f"\nFound {len(X_test)} test images")
    print(f"Authentic images: {np.sum(y_test == 0)}")
    print(f"Forged images: {np.sum(y_test == 1)}")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_proba)
    
    # Create evaluation report
    print("\nCreating evaluation report...")
    create_evaluation_report(y_test, y_pred, y_pred_proba)
    
    print("\nEvaluation complete!")
    return y_test, y_pred, y_pred_proba

if __name__ == '__main__':
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    evaluate_model() 