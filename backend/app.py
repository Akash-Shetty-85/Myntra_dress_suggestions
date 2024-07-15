
from flask import Flask, render_template, jsonify
import json
import random
import os

app = Flask(__name__)

# Ensure predictions.json exists before starting the app
if not os.path.exists('backend/model/predictions.json'):
    from generate_predictions import generate_predictions, load_knn_model
    model = load_knn_model()
    image_dir = 'D:/Projects/Dress_suggestion/data/raw/images'
    generate_predictions(model, image_dir)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_combination')
def get_combination():
    with open('backend/model/predictions.json', 'r') as f:
        combinations = json.load(f)
    combination = random.choice(combinations)
    return jsonify(combination)

if __name__ == '__main__':
    app.run(debug=True)
