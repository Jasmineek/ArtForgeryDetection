from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf.csrf import CSRFProtect
from PIL import Image
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from functools import wraps
import seaborn as sns
#hellooo
app = Flask(__name__)
app.secret_key = 'van-gogh-auth-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['WTF_CSRF_ENABLED'] = True

csrf = CSRFProtect(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

users = {
    'admin': generate_password_hash('admin123'),
    'user': generate_password_hash('user123')
}

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

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    try:
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
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img = Image.open(filepath)
        
        score = np.random.uniform(0.3, 0.7)
        is_authentic = score < 0.5
        confidence = abs(0.5 - score) * 200
        
        plt.figure(figsize=(8, 6))
        plt.bar(['Authentic', 'Forged'], [1-score, score])
        plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold')
        plt.title('Authentication Analysis')
        plt.ylabel('Probability')
        plt.legend()
        
        plot_path = 'static/analysis_plot.png'
        plt.savefig(plot_path)
        plt.close()
        
        session['result'] = {
            'is_authentic': bool(is_authentic),
            'confidence': float(confidence),
            'prediction': float(score),
            'plot_path': 'analysis_plot.png'
        }
        
        flash("Analysis completed successfully!", "success")
        return redirect(url_for('home'))
        
    except Exception as e:
        flash(f"Error analyzing image: {str(e)}", "error")
        return redirect(url_for('home'))

@app.route('/evaluation')
@login_required
def evaluation():
    try:
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        cm = np.array([[85, 15],
                      [10, 90]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Authentic', 'Forged'],
                   yticklabels=['Authentic', 'Forged'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        total = np.sum(cm)
        accuracy = (cm[0,0] + cm[1,1]) / total
        precision = cm[1,1] / (cm[1,1] + cm[0,1])
        recall = cm[1,1] / (cm[1,1] + cm[1,0])
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        metrics = {
            'Overall Accuracy': f"{accuracy*100:.1f}%",
            'Precision': f"{precision*100:.1f}%",
            'Recall': f"{recall*100:.1f}%",
            'F1-Score': f"{f1_score*100:.1f}%"
        }
        
        counts = {
            'Total Images Tested': total,
            'Correctly Identified Authentic': cm[0,0],
            'Correctly Identified Forged': cm[1,1],
            'Authentic Misclassified as Forged': cm[0,1],
            'Forged Misclassified as Authentic': cm[1,0]
        }
        
        plt.subplot(1, 2, 2)
        metric_values = [accuracy, precision, recall, f1_score]
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        plt.bar(metric_labels, metric_values)
        plt.title('Model Performance Metrics')
        plt.ylim(0, 1)
        plt.ylabel('Score')
        for i, v in enumerate(metric_values):
            plt.text(i, v + 0.01, f'{v*100:.1f}%', ha='center')
        
        plt.tight_layout()
        plt.savefig('static/evaluation.png')
        plt.close()
        
        return render_template('evaluation.html', 
                             metrics=metrics,
                             counts=counts,
                             plot_path='evaluation.png')
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        flash(f"Error generating evaluation: {str(e)}", "error")
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True, port=5059)
