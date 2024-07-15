import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

def load_images(image_dir, image_size=(128, 128)):
    data = []
    labels = []
    categories = os.listdir(image_dir)
    for category in categories:
        category_path = os.path.join(image_dir, category)
        if not os.path.isdir(category_path):
            continue
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, image_size)
                data.append(image)
                labels.append(category)
    return np.array(data), np.array(labels)

def train_model(data, labels, model_output_path='backend/model/model.pkl'):
    """Train the KNeighborsClassifier model and save it."""
    # Flatten the images
    X = data.reshape(len(data), -1)  # Flatten the images
    y = labels
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=3)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy}")
    
    # Save the model
    joblib.dump(model, model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    # Use an absolute path directly
    image_dir = 'D:/Projects/Dress_suggestion/data/raw/images'
    print(f"Loading images from: {image_dir}")

    # Check if the directory exists
    if not os.path.exists(image_dir):
        print(f"Directory does not exist: {image_dir}")
    else:
        # Load images and their labels
        data, labels = load_images(image_dir)
        print(f"Loaded {len(data)} images.")
        
        # Train the model and save it
        train_model(data, labels, model_output_path='backend/model/model.pkl')