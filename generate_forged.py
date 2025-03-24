import os
import random
from PIL import Image, ImageEnhance
import numpy as np

def apply_transformations(image):
    """Apply random transformations to create a forged version"""
    # Create a copy of the image
    forged = image.copy()
    
    # Randomly apply different transformations
    transformations = [
        # Adjust brightness
        lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2)),
        # Adjust contrast
        lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2)),
        # Adjust color
        lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2)),
        # Adjust sharpness
        lambda img: ImageEnhance.Sharpness(img).enhance(random.uniform(0.8, 1.2)),
        # Rotate slightly
        lambda img: img.rotate(random.uniform(-5, 5), expand=False),
        # Add noise
        lambda img: add_noise(img)
    ]
    
    # Apply 2-4 random transformations
    num_transformations = random.randint(2, 4)
    selected_transformations = random.sample(transformations, num_transformations)
    
    for transform in selected_transformations:
        forged = transform(forged)
    
    return forged

def add_noise(image):
    """Add random noise to the image"""
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Generate random noise
    noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
    
    # Add noise to image
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Image
    return Image.fromarray(noisy_img)

def generate_forged_paintings():
    """Generate forged paintings from authentic ones"""
    # Create forged directory if it doesn't exist
    forged_dir = 'training_data/forged'
    os.makedirs(forged_dir, exist_ok=True)
    
    # Get list of authentic paintings
    authentic_dir = 'training_data/authentic'
    authentic_files = [f for f in os.listdir(authentic_dir) if f.endswith('.jpg')]
    
    print(f"Generating forged paintings from {len(authentic_files)} authentic paintings...")
    
    for idx, authentic_file in enumerate(authentic_files, 1):
        # Load authentic painting
        authentic_path = os.path.join(authentic_dir, authentic_file)
        authentic_img = Image.open(authentic_path)
        
        # Generate 2 forged versions of each authentic painting
        for version in range(1, 3):
            # Apply transformations
            forged_img = apply_transformations(authentic_img)
            
            # Save forged painting
            forged_filename = f"vangogh_forged_{idx}_{version}_{authentic_file}"
            forged_path = os.path.join(forged_dir, forged_filename)
            forged_img.save(forged_path, 'JPEG', quality=95)
            
            print(f"Generated {forged_filename}")
    
    print("\nGeneration complete!")

if __name__ == '__main__':
    generate_forged_paintings() 