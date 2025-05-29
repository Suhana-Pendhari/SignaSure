import os
import logging
from datetime import datetime
from flask import Flask, request, render_template, flash, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'signasure_secret_key')  # Use environment variable in production

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
try:
    model = load_model('signaSure_model.h5')
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error("Please train the model first using train_model.py")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img):
    """Preprocess the image for prediction."""
    try:
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize
        img = cv2.resize(img, (128, 128))
        
        # Apply adaptive thresholding
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
        
        # Remove small noise
        kernel = np.ones((2,2), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise ValueError(f"Error preprocessing image: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return render_template('index.html')
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return render_template('index.html')
        
        if file and allowed_file(file.filename):
            try:
                # Save the file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Process the image
                img = cv2.imread(filepath)
                if img is None:
                    flash('Error reading image file')
                    return render_template('index.html')
                
                # Preprocess the image
                img = preprocess_image(img)
                
                # Make prediction
                prediction = model.predict(np.expand_dims(img, axis=0))[0][0]
                logger.info(f"Raw prediction: {prediction}")
                
                # Determine result based on prediction
                if prediction > 0.9:
                    result = "Genuine"
                elif prediction < 0.1:
                    result = "Forged"
                elif prediction > 0.7:
                    result = "Likely Genuine"
                elif prediction < 0.3:
                    result = "Likely Forged"
                else:
                    result = "Uncertain"
                
                confidence = f"{abs(prediction - 0.5) * 200:.1f}%"
                logger.info(f"Final result: {result} with confidence {confidence}")
                
                # Clean up
                os.remove(filepath)
                
                # Return only the result box HTML for AJAX requests
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return render_template('_result_box.html', result=result, confidence=confidence)
                
                return render_template('index.html', result=result, confidence=confidence)
                
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                flash('Error processing file')
                return render_template('index.html')
        else:
            flash('Invalid file type. Please upload a PNG, JPG, or JPEG file.')
            return render_template('index.html')
    
    return render_template('index.html')

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File too large. Maximum file size is 16MB.')
    return render_template('index.html'), 413

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {str(error)}")
    flash('An internal server error occurred. Please try again.')
    return render_template('index.html'), 500

if __name__ == '__main__':
    app.run(debug=False)  # Set to False in production 