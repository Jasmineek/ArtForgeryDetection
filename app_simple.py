from flask import Flask, request, render_template, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
from werkzeug.utils import secure_filename
from functools import wraps
from tensorflow.keras.applications.resnet50 import preprocess_input
import traceback

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# User database (in production, use a proper database)
users = {
    'admin': generate_password_hash('admin123'),
    'user': generate_password_hash('user123')
}

# Load the model
model = None
try:
    model = tf.keras.models.load_model('van_gogh_detector.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    try:
        model = tf.keras.models.load_model('van_gogh_detector.keras')
        print("Model loaded successfully from .keras format!")
    except Exception as e:
        print(f"Error loading model from .keras format: {str(e)}")
        model = None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    result = request.args.get('result')
    if result:
        try:
            result = eval(result)  # Convert string back to dict
        except:
            result = None
    return render_template('index.html', result=result)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return 'Invalid username or password'
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

def load_and_preprocess_image(image_path):
    """Load and preprocess an image for the model"""
    try:
        # Load image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        # Convert to array
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        # Expand dimensions
        img_array = np.expand_dims(img_array, axis=0)
        # Preprocess for ResNet50
        img_array = preprocess_input(img_array)
        print(f"Image loaded and preprocessed successfully. Shape: {img_array.shape}")
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        traceback.print_exc()
        return None

def analyze_image(image_path):
    """Analyze the uploaded image"""
    try:
        if model is None:
            print("Error: Model not loaded")
            return None
            
        # Load and preprocess the image
        img = load_and_preprocess_image(image_path)
        if img is None:
            print("Error: Failed to preprocess image")
            return None
        
        # Get prediction
        prediction = model.predict(img, verbose=0)[0][0]
        print(f"Prediction: {prediction}")
        
        # Determine if authentic or forged
        is_authentic = prediction < 0.5
        confidence = (1 - prediction) * 100 if is_authentic else prediction * 100
        
        # Generate analysis text
        analysis_text = f"""
        Analysis Results:
        - Prediction: {'Authentic' if is_authentic else 'Forged'}
        - Confidence: {confidence:.2f}%
        - Raw Score: {prediction:.4f}
        """
        
        # Create a simple visualization
        plt.figure(figsize=(10, 4))
        plt.bar(['Authentic', 'Forged'], [1-prediction, prediction])
        plt.title('Prediction Confidence')
        plt.ylabel('Probability')
        
        # Save the plot
        plot_path = os.path.join('static', 'analysis_plot.png')
        plt.savefig(plot_path)
        plt.close()
        
        return {
            'is_authentic': is_authentic,
            'confidence': confidence,
            'analysis_text': analysis_text,
            'plot_path': 'analysis_plot.png'
        }
        
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        traceback.print_exc()
        return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if 'image' not in request.files:
        print("No image uploaded")
        return 'No image uploaded', 400
    
    file = request.files['image']
    if file.filename == '':
        print("No image selected")
        return 'No image selected', 400
    
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"Image saved to {filepath}")
            
            # Analyze the image
            result = analyze_image(filepath)
            
            if result:
                print("Analysis successful:", result)
                return redirect(url_for('home', result=str(result)))
            else:
                print("Analysis failed")
                return 'Error analyzing image', 500
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            traceback.print_exc()
            return f'Error analyzing image: {str(e)}', 500
    
    print("Invalid file type")
    return 'Invalid file type', 400

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5055) 