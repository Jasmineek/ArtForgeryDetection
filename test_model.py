Create an improved training script with these enhancementsimport tensorflow as tf
import numpy as np
from PIL import Image
import os

    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def test_painting(model_path, image_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the image
    img_array = load_and_preprocess_image(image_path)
    if img_array is None:
        return
    
    # Make prediction
    prediction = model.predict(img_array)[0][0]
    
    # Print results
    print(f"\nTesting image: {os.path.basename(image_path)}")
    print(f"Prediction score: {prediction:.3f}")
    print(f"Classification: {'Forged' if prediction > 0.5 else 'Authentic'}")
    print(f"Confidence: {abs(prediction - 0.5) * 2:.1%}")

def main():
    model_path = 'van_gogh_detector.h5'
    
    # Test authentic paintings
    print("\nTesting Authentic Van Gogh Paintings:")
    authentic_dir = 'data/authentic'
    for image_file in os.listdir(authentic_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(authentic_dir, image_file)
            test_painting(model_path, image_path)
    
    # Test forged paintings
    print("\nTesting Forged Paintings:")
    forged_dir = 'data/forged'
    # Test first 5 forged paintings
    for image_file in sorted(os.listdir(forged_dir))[:5]:
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(forged_dir, image_file)
            test_painting(model_path, image_path)

if __name__ == "__main__":
    main() 