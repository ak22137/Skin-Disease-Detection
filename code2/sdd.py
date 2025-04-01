import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator

Categories = ['VI-shingles', 'VI-chickenpox', 'BA- cellulitis', 'FU-athlete-foot', 
              'BA-impetigo', 'FU-nail-fungus', 'FU-ringworm', 'PA-cutaneous-larva-migrans']

test_dir = r'C:\Users\hp\Desktop\Project\code2\skin-disease-datasaet\test_set'
train_dir = r'C:\Users\hp\Desktop\Project\code2\skin-disease-datasaet\train_set'
img_size = (224, 224)

def preprocess_image(image):
    if image is None:
        raise ValueError("Image is None")
    
    # Ensure image is in uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Resize
    image = cv2.resize(image, img_size)
    
    # Contrast Enhancement (CLAHE)
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        print(f"Warning: CLAHE enhancement failed, using original image. Error: {str(e)}")
    
    # Convert to float32 and normalize to [0, 1]
    image = image.astype(np.float32)
    
    # Apply ResNet50 preprocessing
    image = resnet_preprocess(image)
    return image

def create_data(directory):
    data = []
    labels = []
    for category_idx, category in enumerate(Categories):
        path = os.path.join(directory, category)
        if not os.path.exists(path):
            print(f"Warning: Directory not found: {path}")
            continue
            
        print(f"Processing category: {category}")
        category_count = 0
        
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img_array = cv2.imread(img_path)
                
                if img_array is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                
                processed_img = preprocess_image(img_array)
                data.append(processed_img)
                labels.append(category_idx)
                category_count += 1
                
            except Exception as e:
                print(f"Error processing {img_name} in category {category}: {str(e)}")
                
        print(f"Processed {category_count} images for {category}")
    
    if not data:
        raise ValueError("No images were successfully loaded and processed")
    
    return np.array(data), np.array(labels)

# Load and preprocess data
print("Loading training data...")
train_data, train_labels = create_data(train_dir)
print(f"Training data shape: {train_data.shape}")

print("\nLoading test data...")
test_data, test_labels = create_data(test_dir)
print(f"Test data shape: {test_data.shape}")

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Feature Extraction using ResNet50
print("\nExtracting features using ResNet50...")
base_model = ResNet50(weights='imagenet', include_top=False, 
                     input_shape=(img_size[0], img_size[1], 3))

train_features = base_model.predict(train_data, batch_size=32, verbose=1)
test_features = base_model.predict(test_data, batch_size=32, verbose=1)

# Flatten features
train_features_flat = train_features.reshape(train_features.shape[0], -1)
test_features_flat = test_features.reshape(test_features.shape[0], -1)

# Train SVM classifier
print("\nTraining SVM classifier...")
svm = SVC(kernel='rbf', C=1.0, probability=True)
svm.fit(train_features_flat, train_labels)

# Predictions and Evaluation
predictions = svm.predict(test_features_flat)
print("\nClassification Report:")
print(classification_report(test_labels, predictions, target_names=Categories))

print("\nConfusion Matrix:")
print(confusion_matrix(test_labels, predictions))

# Visualize results
plt.figure(figsize=(15, 5))
for i in range(min(5, len(test_data))):
    plt.subplot(1, 5, i+1)
    # Convert back to RGB for display
    img = test_data[i]
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize to [0,1]
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    predicted_label = Categories[predictions[i]]
    true_label = Categories[test_labels[i]]
    plt.title(f"Pred: {predicted_label.split('-')[0]}\nTrue: {true_label.split('-')[0]}", 
              fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.show()

import pickle
import joblib

# Save SVM model
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)

# Save ResNet50 model
base_model.save('resnet50_model.h5')