import joblib
import cv2
import numpy as np

def load_model(model_path='model.pkl'):
    return joblib.load(model_path)

def predict_combination(model, image_path, image_size=(128, 128)):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, image_size)
        image = image.flatten().reshape(1, -1)  # Flatten and reshape for prediction
        prediction = model.predict(image)
        return prediction
    else:
        return ["Invalid image"]

model = load_model()
