# SignaSure - AI-Powered Signature Verification System

SignaSure is a deep learning-based signature verification system that can distinguish between genuine and forged signatures. The system uses a Convolutional Neural Network (CNN) to analyze signature images and determine their authenticity with high accuracy.

## Features

- Real-time signature verification
- User-friendly web interface
- Drag-and-drop signature upload
- Confidence score for predictions
- Support for PNG, JPG, and JPEG formats
- Adaptive preprocessing for better accuracy
- AJAX-based form submission (no page refresh)
- Real-time image preview
- Loading indicators
- Enhanced error handling

## Model Performance

- Training Accuracy: 93.71%
- Validation Accuracy: 92.66%
- False Positive Rate: < 5%
- False Negative Rate: < 5%
- Average prediction time: < 1 second

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Suhana-Pendhari/SignaSure.git
cd SignaSure
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
SignaSure/
├── app.py              # Flask web application
├── train_model.py      # Model training script
├── templates/          # HTML templates
│   ├── index.html     # Main web interface
│   └── _result_box.html # Result display component
├── dataset/           # Training data
│   ├── genuine/       # Genuine signatures
│   └── forged/        # Forged signatures
├── uploads/           # Temporary storage
├── requirements.txt   # Python dependencies
├── model.md          # Model documentation
└── README.md         # Project documentation
```

## Dataset

The dataset used in this project is not included in the repository due to size limitations. To use this project, you'll need to:

1. Create a `dataset` directory in the project root
2. Download the dataset from [Google Drive Link] (to be added)
3. Extract the dataset into the `dataset` directory with the following structure:
```
dataset/
├── genuine/
│   ├── original_*.png
│   └── sample_*.png
└── forged/
    └── forged_*.png
```

## Usage

1. Train the model:
```bash
python train_model.py
```
This will create a trained model file (`signaSure_model.h5`).

2. Start the web application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

4. Upload a signature image using the web interface:
   - Drag and drop or click to select
   - Supported formats: PNG, JPG, JPEG
   - Maximum size: 16MB

## How It Works

1. **Preprocessing**:
   - Converts image to grayscale
   - Resizes to 128x128 pixels
   - Applies adaptive thresholding
   - Removes noise
   - Normalizes pixel values

2. **Model Architecture**:
   - Three convolutional blocks with batch normalization
   - MaxPooling layers for dimensionality reduction
   - Dense layers with dropout for classification
   - Binary output (genuine/forged)

3. **Training**:
   - Uses balanced dataset of genuine and forged signatures
   - Data augmentation for better generalization
   - Early stopping to prevent overfitting
   - Model checkpointing to save best weights

4. **Prediction**:
   - Real-time image processing
   - Confidence-based classification
   - Multiple result categories:
     - Genuine (> 0.9)
     - Forged (< 0.1)
     - Likely Genuine (0.7-0.9)
     - Likely Forged (0.3-0.4)
     - Uncertain (0.4-0.7)

## Technical Details

- **Framework**: TensorFlow/Keras
- **Web Framework**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **Image Processing**: OpenCV
- **Data Augmentation**: Keras ImageDataGenerator

## Recent Improvements

1. **Model Enhancements**:
   - Improved prediction thresholds
   - Enhanced preprocessing
   - Better normalization

2. **UI/UX Improvements**:
   - AJAX-based form submission
   - Real-time image preview
   - Drag-and-drop support
   - Loading indicators
   - Better error handling
   - Confidence score display

## Future Improvements

1. **Model Enhancements**:
   - Transfer learning with pre-trained models
   - Ensemble methods
   - Real-time training with user feedback

2. **Feature Additions**:
   - Multiple signature comparison
   - Batch processing
   - API endpoints
   - User authentication
   - Signature database

3. **Performance Optimizations**:
   - GPU acceleration
   - Model quantization
   - Caching mechanisms

4. **UI/UX Improvements**:
   - Mobile responsiveness
   - Dark mode
   - Advanced visualization
   - User preferences

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow and Keras for deep learning capabilities
- Flask for web framework
- OpenCV for image processing
- Bootstrap for frontend styling

## Contact

For any questions or suggestions, please open an issue in the GitHub repository.