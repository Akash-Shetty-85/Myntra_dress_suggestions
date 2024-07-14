import os
import cv2
import numpy as np
import joblib
import json

def load_knn_model():
    """Load the KNN model from a file."""
    return joblib.load('backend/model/model.pkl')

def generate_predictions(model, image_dir, output_file='backend/model/predictions.json'):
    """Generate predictions for specific dress combinations and save them."""
    categories = os.listdir(image_dir)
    all_images = []
    all_labels = []
    image_paths = []
    
    # Dictionary to store images by their category
    images_by_category = {category: [] for category in categories if category in ['tshirts', 'shorts', 'pants', 'shirts']}
    
    for category in images_by_category.keys():
        category_path = os.path.join(image_dir, category)
        if not os.path.isdir(category_path):
            continue
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, (128, 128))  # Same size as used for training
                all_images.append(image)
                all_labels.append(category)
                image_paths.append(image_path)
                images_by_category[category].append(image_path)
    
    all_images = np.array(all_images).reshape(len(all_images), -1)  # Flatten the images for KNN
    predictions = model.predict(all_images)

    # Create a dictionary for image paths and their predicted labels
    path_label_dict = dict(zip(image_paths, predictions))

    # Create a directory for saving combined images
    output_images_dir = 'backend/static/combined_images'
    os.makedirs(output_images_dir, exist_ok=True)

    # List to store the generated combinations
    combinations_list = []

    # Define the valid pairs
    valid_pairs = [
        ('tshirts', 'shorts'),
        ('tshirts', 'pants'),
        ('shirts', 'pants')
    ]

    for (cat1, cat2) in valid_pairs:
        cat1_images = images_by_category[cat1]
        cat2_images = images_by_category[cat2]
        for item1_path in cat1_images:
            for item2_path in cat2_images:
                # Load and prepare images for the combination
                item1_image = cv2.imread(item1_path)
                item2_image = cv2.imread(item2_path)
                item1_image = cv2.resize(item1_image, (128, 128))
                item2_image = cv2.resize(item2_image, (128, 128))

                # Combine images side-by-side
                combined_image = np.hstack((item1_image, item2_image))
                combined_image_path = os.path.join(output_images_dir, f"{os.path.basename(item1_path)}_with_{os.path.basename(item2_path)}.jpg")
                cv2.imwrite(combined_image_path, combined_image)

                # Add combination info and image path to the list
                combinations_list.append({
                    'combo': f"{os.path.basename(item1_path)} with {os.path.basename(item2_path)}",
                    'image': f"/static/combined_images/{os.path.basename(combined_image_path)}"
                })

    # Save predictions to a file
    with open(output_file, 'w') as f:
        json.dump(combinations_list, f)

    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    image_dir = 'D:/Projects/Dress_suggestion/data/raw/images'
    model = load_knn_model()
    generate_predictions(model, image_dir)
