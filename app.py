from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf.csrf import CSRFProtect
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import traceback
from functools import wraps
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'van-gogh-auth-secret-key-2024'  # Set a proper secret key
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['WTF_CSRF_ENABLED'] = True

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# User database (in production, use a proper database)
users = {
    'admin': generate_password_hash('admin123'),
    'user': generate_password_hash('user123')
}

# Load the model
model = None
try:
    model = tf.keras.models.load_model('van_gogh_detector.keras')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")

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
    return render_template('presentation.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            password = request.form.get('password')
            
            print(f"Login attempt for user: {username}")
            
            if not username or not password:
                flash('Please enter both username and password')
                return render_template('login.html')
            
            if username in users and check_password_hash(users[username], password):
                session['username'] = username
                print(f"Successful login for user: {username}")
                return redirect(url_for('home'))
            else:
                print(f"Failed login attempt for user: {username}")
                flash('Invalid username or password')
        except Exception as e:
            print(f"Error during login: {str(e)}")
            flash('An error occurred during login. Please try again.')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def preprocess_image(image):
    """Preprocess the image for model input"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image to match ResNet50 input size
        image = image.resize((224, 224))
        
        # Convert to numpy array and preprocess
        img_array = np.array(image)
        img_array = preprocess_input(img_array)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    try:
        print("Received analyze request")
        if 'image' not in request.files:
            flash("No image file provided")
            return redirect(url_for('home'))
        
        file = request.files['image']
        if file.filename == '':
            flash("No selected file")
            return redirect(url_for('home'))
        
        if not allowed_file(file.filename):
            flash("Invalid file type. Please upload a JPG or PNG image.")
            return redirect(url_for('home'))
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"Saved file to: {filepath}")
        
        # Read and preprocess the image
        image = Image.open(filepath)
        img_array = preprocess_image(image)
        print("Image preprocessed successfully")
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)[0][0]
        print(f"Prediction: {prediction}")
        
        # Calculate confidence
        confidence = abs(prediction - 0.5) * 200  # Convert to percentage
        print(f"Confidence: {confidence}%")
        
        # Create visualization
        plt.figure(figsize=(8, 6))
        plt.bar(['Forged', 'Authentic'], [prediction, 1-prediction])
        plt.title('Prediction Confidence')
        plt.ylabel('Probability')
        
        # Save plot
        plot_path = 'static/analysis_plot.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to: {plot_path}")
        
        # Store result in session
        session['result'] = {
            'is_authentic': bool(prediction < 0.5),  # Convert numpy bool_ to Python bool
            'confidence': float(confidence),
            'prediction': float(prediction),
            'plot_path': 'analysis_plot.png'
        }
        print("Analysis completed successfully")
        
        return redirect(url_for('home'))
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print(traceback.format_exc())
        flash(f"Error analyzing image: {str(e)}")
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True, port=5055)
