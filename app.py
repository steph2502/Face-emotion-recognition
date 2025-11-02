import os
import cv2
import sqlite3
import numpy as np
import requests
from flask import Flask, render_template, request
from keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = 'face_emotionModel.h5'
MODEL_URL = os.getenv('MODEL_URL')  # Add this in your Render environment settings

# -------------------------------
# Download model from external URL
# -------------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        if not MODEL_URL:
            raise RuntimeError("MODEL_URL not set in environment variables.")
        print("Downloading model from external URL...")
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully!")

download_model()
model = load_model(MODEL_PATH)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# -------------------------------
# Database setup
# -------------------------------
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            department TEXT,
            image_path TEXT,
            emotion TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# -------------------------------
# Emotion prediction function
# -------------------------------
def predict_emotion(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48))
    gray = gray.astype('float32') / 255.0
    gray = np.expand_dims(gray, axis=(0, -1))
    preds = model.predict(gray)
    emotion_index = np.argmax(preds[0])
    return emotion_labels[emotion_index]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form['name']
    email = request.form['email']
    department = request.form['department']
    file = request.files['photo']

    if not file:
        return "No image uploaded."

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    emotion = predict_emotion(file_path)

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('INSERT INTO students (name, email, department, image_path, emotion) VALUES (?, ?, ?, ?, ?)',
              (name, email, department, file_path, emotion))
    conn.commit()
    conn.close()

    responses = {
        'Happy': "You are smiling. You look happy today 😊",
        'Sad': "You look sad. Hope everything is okay 💙",
        'Angry': "You seem upset. Take a deep breath 😤",
        'Fear': "You look scared. Don’t worry, you got this 😨",
        'Disgust': "You look disgusted. Something bothering you? 🤢",
        'Neutral': "You look calm and neutral 😐",
        'Surprise': "Wow! You look surprised 😲"
    }

    message = responses.get(emotion, "Emotion detected.")
    return f"<h2>{message}</h2><p>Detected emotion: {emotion}</p><a href='/'>Try again</a>"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
