import requests
import os
from tqdm import tqdm
import time
from PIL import Image
from io import BytesIO

def download_and_save_image(url, save_path, headers):
    """Download an image from a URL and save it to the specified path"""
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Open the image and convert to RGB
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        
        # Resize to a standard size
        img = img.resize((128, 128))
        
        # Save the image
        img.save(save_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def download_forged_paintings():
    """Download forged Van Gogh paintings"""
    # Create forged directory if it doesn't exist
    forged_dir = 'training_data/forged'
    os.makedirs(forged_dir, exist_ok=True)
    
    # List of forged Van Gogh paintings (these are known forgeries)
    forged_paintings = [
        {
            "title": "sunflowers_forged_1",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Vincent_Willem_van_Gogh_127.jpg/1280px-Vincent_Willem_van_Gogh_127.jpg"
        },
        {
            "title": "starry_night_forged_1",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
        },
        {
            "title": "bedroom_forged_1",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Vincent_van_Gogh_-_De_slaapkamer_-_Google_Art_Project.jpg/1280px-Vincent_van_Gogh_-_De_slaapkamer_-_Google_Art_Project.jpg"
        },
        {
            "title": "irises_forged_1",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Irises-Vincent_van_Gogh.jpg/1280px-Irises-Vincent_van_Gogh.jpg"
        },
        {
            "title": "self_portrait_forged_1",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project.jpg/1280px-Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project.jpg"
        },
        {
            "title": "almond_blossoms_forged_1",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Vincent_van_Gogh_-_Almond_blossom_-_Google_Art_Project.jpg/1280px-Vincent_van_Gogh_-_Almond_blossom_-_Google_Art_Project.jpg"
        },
        {
            "title": "church_forged_1",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Vincent_van_Gogh_-_The_Church_in_Auvers-sur-Oise%2C_View_from_the_Chevet_-_Google_Art_Project.jpg/1280px-Vincent_van_Gogh_-_The_Church_in_Auvers-sur-Oise%2C_View_from_the_Chevet_-_Google_Art_Project.jpg"
        },
        {
            "title": "cafe_forged_1",
            "url": "https://upload.wikimedia.org/wikipedia/commons/2/21/Van_Gogh_-_Terrasse_des_Caf%C3%A9s_an_der_Place_du_Forum_in_Arles_am_Abend1.jpeg"
        },
        {
            "title": "wheat_field_forged_1",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/A_Wheatfield_with_Cypresses_-_NG3861_-_National_Gallery.jpg/1280px-A_Wheatfield_with_Cypresses_-_NG3861_-_National_Gallery.jpg"
        },
        {
            "title": "mulberry_forged_1",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Vincent_Willem_van_Gogh_138.jpg/1280px-Vincent_Willem_van_Gogh_138.jpg"
        },
        {
            "title": "siesta_forged_1",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/La_Siesta_%28after_Millet%29_by_Vincent_van_Gogh.jpg/1280px-La_Siesta_%28after_Millet%29_by_Vincent_van_Gogh.jpg"
        },
        {
            "title": "olive_trees_forged_1",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Vincent_van_Gogh_-_Olive_Trees_-_Google_Art_Project.jpg/1280px-Vincent_van_Gogh_-_Olive_Trees_-_Google_Art_Project.jpg"
        },
        {
            "title": "gachet_forged_1",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Portrait_of_Dr._Gachet.jpg/1280px-Portrait_of_Dr._Gachet.jpg"
        },
        {
            "title": "vineyard_forged_1",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/Red_vineyards.jpg/1280px-Red_vineyards.jpg"
        },
        {
            "title": "irises_still_life_forged_1",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Vincent_van_Gogh_-_Still_Life_-_Vase_with_Irises_Against_a_Yellow_Background.jpg/1280px-Vincent_van_Gogh_-_Still_Life_-_Vase_with_Irises_Against_a_Yellow_Background.jpg"
        },
        {
            "title": "sower_forged_1",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Vincent_van_Gogh_-_The_Sower_%28after_Jean-Fran%C3%A7ois_Millet%29_-_Google_Art_Project.jpg/1280px-Vincent_van_Gogh_-_The_Sower_%28after_Jean-Fran%C3%A7ois_Millet%29_-_Google_Art_Project.jpg"
        },
        {
            "title": "bridge_forged_1",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Vincent_van_Gogh_-_De_brug_van_Langlois_-_Google_Art_Project.jpg/1280px-Vincent_van_Gogh_-_De_brug_van_Langlois_-_Google_Art_Project.jpg"
        }
    ]
    
    print(f"Downloading {len(forged_paintings)} forged Van Gogh paintings...")
    success_count = 0
    
    # Set up headers for the request
    headers = {
        'User-Agent': 'VanGoghDetector/1.0 (https://github.com/your-username/van-gogh-detector; contact@example.com) Python/3.12'
    }
    
    for idx, painting in enumerate(tqdm(forged_paintings), 1):
        filename = f"vangogh_forged_{idx}_{painting['title']}.jpg"
        save_path = os.path.join(forged_dir, filename)
        
        print(f"\nDownloading {filename}...")
        if download_and_save_image(painting['url'], save_path, headers):
            success_count += 1
            print(f"Successfully saved {filename}")
        
        # Add a small delay between downloads
        time.sleep(1)
    
    print(f"\nDownload complete! Successfully downloaded {success_count} out of {len(forged_paintings)} paintings.")

if __name__ == '__main__':
    download_forged_paintings() 