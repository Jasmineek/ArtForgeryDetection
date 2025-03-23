from flask import Flask, request, render_template, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from werkzeug.utils import secure_filename
from functools import wraps
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)
app.secret_key = 'van-gogh-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Simple user database
users = {
    'admin': generate_password_hash('admin123'),
    'user': generate_password_hash('user123')
}

# Load the model
model = None
try:
    model = tf.keras.models.load_model('van_gogh_detector.keras')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
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
    return render_template('presentation.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

def analyze_image(image_path):
    """Analyze the uploaded image"""
    try:
        if model is None:
            return {'error': 'Model not loaded'}
            
        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Get prediction
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # Determine if authentic or forged
        is_authentic = prediction < 0.5
        confidence = (1 - prediction) * 100 if is_authentic else prediction * 100
        
        # Create visualization
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
            'prediction': prediction,
            'plot_path': 'analysis_plot.png'
        }
        
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        return {'error': str(e)}

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    if 'image' not in request.files:
        flash('No image uploaded')
        return redirect(url_for('home'))
    
    file = request.files['image']
    if file.filename == '':
        flash('No image selected')
        return redirect(url_for('home'))
    
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyze the image
            result = analyze_image(filepath)
            
            if 'error' in result:
                flash(f'Error analyzing image: {result["error"]}')
            else:
                session['result'] = result
                session['uploaded_image'] = filename
                
        except Exception as e:
            flash(f'Error: {str(e)}')
    else:
        flash('Invalid file type')
    
    return redirect(url_for('home'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5055) 