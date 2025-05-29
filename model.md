# SignaSure Model Documentation

## Model Architecture

### CNN Architecture
- Input Layer: 128x128 grayscale images
- Three Convolutional Blocks:
  - Block 1: 32 filters, 3x3 kernel, ReLU activation
  - Block 2: 64 filters, 3x3 kernel, ReLU activation
  - Block 3: 128 filters, 3x3 kernel, ReLU activation
- MaxPooling layers after each block
- Batch Normalization for improved training stability
- Dense layers with dropout (0.5) for classification
- Binary output (genuine/forged)

### Image Preprocessing
- Grayscale conversion
- Resize to 128x128 pixels
- Adaptive thresholding
- Noise removal
- Pixel value normalization (0-1)

## Training Process

### Dataset
- Balanced dataset with genuine and forged signatures
- Data augmentation for better generalization
- Training/Validation split: 80/20

### Training Parameters
- Optimizer: Adam
- Loss Function: Binary Cross-Entropy
- Batch Size: 32
- Epochs: 50
- Early Stopping with patience=5
- Model checkpointing for best weights

## Recent Improvements

### Model Enhancements
1. Improved prediction thresholds:
   - Genuine: > 0.9
   - Forged: < 0.1
   - Likely Genuine: 0.7-0.9
   - Likely Forged: 0.3-0.4
   - Uncertain: 0.4-0.7

2. Enhanced preprocessing:
   - Adaptive thresholding
   - Noise removal
   - Better normalization

### UI/UX Improvements
1. AJAX-based form submission
2. Real-time image preview
3. Drag-and-drop support
4. Loading indicators
5. Better error handling
6. Confidence score display

## Performance Metrics

### Model Performance
- Training Accuracy: 93.71%
- Validation Accuracy: 92.66%
- False Positive Rate: < 5%
- False Negative Rate: < 5%

### System Performance
- Average prediction time: < 1 second
- Support for multiple image formats (PNG, JPG, JPEG)
- Maximum file size: 16MB

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
└── README.md         # Project documentation
```

## Usage Instructions

1. Training the Model:
```bash
python train_model.py
```

2. Running the Application:
```bash
python app.py
```

3. Access the web interface at http://localhost:5000

4. Upload a signature image:
   - Drag and drop or click to select
   - Supported formats: PNG, JPG, JPEG
   - Maximum size: 16MB

5. View Results:
   - Genuine/Forged classification
   - Confidence score
   - Visual indicators

## Future Improvements

1. Model Enhancements:
   - Transfer learning with pre-trained models
   - Ensemble methods
   - Real-time training with user feedback

2. Feature Additions:
   - Multiple signature comparison
   - Batch processing
   - API endpoints
   - User authentication
   - Signature database

3. Performance Optimizations:
   - GPU acceleration
   - Model quantization
   - Caching mechanisms

4. UI/UX Improvements:
   - Mobile responsiveness
   - Dark mode
   - Advanced visualization
   - User preferences

## Technical Details

- **Framework**: TensorFlow/Keras
- **Web Framework**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **Image Processing**: OpenCV
- **Data Augmentation**: Keras ImageDataGenerator

### TensorFlow Optimizations
- Using oneDNN custom operations for improved performance
- CPU instruction optimizations enabled (SSE3, SSE4.1, SSE4.2, AVX, AVX2, AVX512F, AVX512_VNNI, FMA)
- Note: Some numerical results may vary slightly due to floating-point round-off errors from different computation orders
- To disable oneDNN optimizations, set environment variable: `TF_ENABLE_ONEDNN_OPTS=0`

### Model Format
- Model saved in HDF5 format (signaSure_model.h5)
- Note: Future versions will use the native Keras format (.keras) for better compatibility

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License. 