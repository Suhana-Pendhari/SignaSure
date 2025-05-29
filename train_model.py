import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import cv2
from PIL import Image
import random
import logging
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_synthetic_signature():
    """Generate a synthetic signature image."""
    # Create a blank image
    img = np.ones((128, 128), dtype=np.uint8) * 255
    
    # Generate random points for the signature
    num_points = random.randint(50, 100)
    points = []
    x, y = random.randint(20, 40), random.randint(20, 40)
    points.append((x, y))
    
    for _ in range(num_points):
        # Add some randomness to the movement
        x += random.randint(-5, 5)
        y += random.randint(-5, 5)
        # Keep within bounds
        x = max(10, min(x, 118))
        y = max(10, min(y, 118))
        points.append((x, y))
    
    # Draw the signature
    for i in range(len(points)-1):
        cv2.line(img, points[i], points[i+1], 0, random.randint(1, 3))
    
    # Add some noise
    noise = np.random.normal(0, 10, (128, 128)).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # Apply some random transformations
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        center = (64, 64)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (128, 128))
    
    return img

def preprocess_image(img):
    """Preprocess the image for training or prediction."""
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

def load_real_signatures(data_dir):
    """Load real signature images from the dataset directory."""
    images = []
    labels = []
    
    # Load genuine signatures
    genuine_dir = os.path.join(data_dir, 'genuine')
    for img_path in os.listdir(genuine_dir):
        full_path = os.path.join(genuine_dir, img_path)
        try:
            img = cv2.imread(full_path)
            if img is not None:
                img = preprocess_image(img)
                images.append(img)
                labels.append(1)  # 1 for genuine
                logger.info(f"Loaded genuine signature: {img_path}")
        except Exception as e:
            logger.error(f"Error loading {img_path}: {str(e)}")
    
    # Load forged signatures
    forged_dir = os.path.join(data_dir, 'forged')
    for img_path in os.listdir(forged_dir):
        full_path = os.path.join(forged_dir, img_path)
        try:
            img = cv2.imread(full_path)
            if img is not None:
                img = preprocess_image(img)
                images.append(img)
                labels.append(0)  # 0 for forged
                logger.info(f"Loaded forged signature: {img_path}")
        except Exception as e:
            logger.error(f"Error loading {img_path}: {str(e)}")
    
    return np.array(images), np.array(labels)

def create_dataset(num_samples=1000, data_dir='dataset'):
    """Create a dataset using both synthetic and real signatures."""
    print("Creating dataset...")
    X = []
    y = []
    
    # Generate synthetic signatures
    for _ in range(num_samples):
        img = create_synthetic_signature()
        img = preprocess_image(img)
        X.append(img)
        y.append(1)
    
    for _ in range(num_samples):
        img = create_synthetic_signature()
        # Add more distortion for forged signatures
        img = cv2.GaussianBlur(img, (3,3), 0)
        img = preprocess_image(img)
        X.append(img)
        y.append(0)
    
    # Load real signatures
    real_X, real_y = load_real_signatures(data_dir)
    X.extend(real_X)
    y.extend(real_y)
    
    return np.array(X), np.array(y)

def create_model():
    """Create and compile the CNN model."""
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

def train_model():
    # Load real signatures
    logger.info("Loading real signatures...")
    X, y = load_real_signatures('dataset')
    
    if len(X) == 0:
        logger.error("No signature images found in dataset directory!")
        return
    
    logger.info(f"Loaded {len(X)} signature images")
    logger.info(f"Genuine signatures: {np.sum(y == 1)}")
    logger.info(f"Forged signatures: {np.sum(y == 0)}")
    
    # Reshape for CNN input
    X = X.reshape(-1, 128, 128, 1)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create data generator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest',
        horizontal_flip=False,
        vertical_flip=False
    )
    
    # Create and compile model
    logger.info("Creating model...")
    model = create_model()
    
    # Train the model
    logger.info("Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    # Save the model
    model.save('signature_model.h5')
    
    return model, history

if __name__ == "__main__":
    model, history = train_model()
    
    # Print final metrics
    print("\nFinal Training Accuracy:", history.history['accuracy'][-1])
    print("Final Validation Accuracy:", history.history['val_accuracy'][-1]) 