import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications.resnet50 import preprocess_input

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image"""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Use ResNet50's preprocessing
    return img_array

def test_model():
    """Test the model on authentic and forged images"""
    # Load the model
    print("Loading model...")
    model = tf.keras.models.load_model('van_gogh_detector.keras')  # Changed to .keras format
    print("Model loaded successfully!")
    
    # Test authentic images
    print("\nTesting authentic images (should be close to 0):")
    authentic_dir = 'data/authentic'
    for filename in os.listdir(authentic_dir):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(authentic_dir, filename)
            img = load_and_preprocess_image(img_path)
            prediction = model.predict(img, verbose=0)[0][0]  # Added verbose=0 to reduce output
            print(f"{filename}: {prediction:.4f}")
    
    # Test forged images
    print("\nTesting forged images (should be close to 1):")
    forged_dir = 'data/forged'
    for filename in os.listdir(forged_dir):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(forged_dir, filename)
            img = load_and_preprocess_image(img_path)
            prediction = model.predict(img, verbose=0)[0][0]  # Added verbose=0 to reduce output
            print(f"{filename}: {prediction:.4f}")

if __name__ == '__main__':
    test_model() 