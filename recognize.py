from flask import Flask, request, render_template, jsonify
import json
import os
from google.cloud import vision

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vision-api-key.json'
client = vision.ImageAnnotatorClient()

def load_food_names(file_name):
    with open(file_name, 'r') as file:
        return {line.strip().lower() for line in file}

known_foods = load_food_names('known_foods.dict')

def detect_ingredients(image_data):
    image = vision.Image(content=image_data)
    response = client.label_detection(image=image)
    
    if response.error.message:
        print(f"Error from Vision API: {response.error.message}")
        return []

    ingredients = []
    for label in response.label_annotations:
        desc = label.description.lower()
        score = label.score
        if desc in known_foods and score > 0.7:
            ingredients.append(desc)
    return ingredients

app = Flask(__name__)

def find_desserts(fruits):
    """Find possible desserts based on detected fruits"""
    with open('fruit_recipes.json') as f:
        fruit_recipes = json.load(f)
    matched_desserts = set()
    for fruit in fruits:
        if fruit in fruit_recipes:
            matched_desserts.update(fruit_recipes[fruit])
    return list(matched_desserts)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    image = file.read()
    
    # Detect fruits using Vision API
    detected_fruits = detect_ingredients(image)
    
    # Find matching desserts
    possible_desserts = find_desserts(detected_fruits)
    
    # Return the results as JSON
    return jsonify({'detected_fruits': detected_fruits, 'possible_desserts': possible_desserts})

if __name__ == '__main__':
    app.run(debug=True)
