from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import json
import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from transformers import BartModel, BartTokenizer
from torchvision import models
import pandas as pd
import requests as req
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the tokenizer
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)

# Define image transformations for inference
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the CSV file to get the label list
text_df = pd.read_csv("data.csv")
label_list = text_df['label'].unique().tolist()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "multimodal_model.pth"

# Load the pre-trained ResNet18 model and replace its fc layer with Identity
image_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
image_model.fc = nn.Identity()

# Create the MultiModalClassifier using the modified image model
class MultiModalClassifier(nn.Module):
    def __init__(self, text_model, image_model, text_feat_dim, image_feat_dim, hidden_dim, num_classes):
        super(MultiModalClassifier, self).__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.text_fc = nn.Linear(text_feat_dim, hidden_dim)
        self.image_fc = nn.Linear(image_feat_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text_input=None, image_input=None):
        features = None

        if text_input is not None:
            text_input_filtered = {k: v for k, v in text_input.items() if k != "labels"}
            text_outputs = self.text_model(**text_input_filtered)
            pooled_text = text_outputs.last_hidden_state.mean(dim=1)
            text_features = self.text_fc(pooled_text)
            features = text_features if features is None else features + text_features

        if image_input is not None:
            image_features = self.image_model(image_input)
            image_features = self.image_fc(image_features)
            features = image_features if features is None else features + image_features

        if (text_input is not None) and (image_input is not None):
            features = features / 2

        logits = self.classifier(features)
        return logits

model = MultiModalClassifier(
    text_model=BartModel.from_pretrained(model_name),
    image_model=image_model,
    text_feat_dim=768,
    image_feat_dim=512,
    hidden_dim=512,
    num_classes=len(label_list)
)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}


# ─────────────────────────────────────────────
# Inference Functions
# ─────────────────────────────────────────────

def inference_text(model, tokenizer, text, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, padding="max_length", truncation=True,
                         max_length=max_length, return_tensors="pt")
    for key in encoding:
        encoding[key] = encoding[key].to(device)
    with torch.no_grad():
        logits = model(text_input=encoding, image_input=None)
    pred_id = torch.argmax(logits, dim=1).item()
    return id2label[pred_id], text

def inference_image(model, image_path, transform, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(text_input=None, image_input=image)
    pred_id = torch.argmax(logits, dim=1).item()
    return id2label[pred_id], image_path

def inference_both(model, tokenizer, text, image_path, transform, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, padding="max_length", truncation=True,
                         max_length=max_length, return_tensors="pt")
    for key in encoding:
        encoding[key] = encoding[key].to(device)
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(text_input=encoding, image_input=image)
    pred_id = torch.argmax(logits, dim=1).item()
    return id2label[pred_id], text, image_path

def generate_report(predicted_label, text, image_path):
    return f"""
    Thyroid Disease Prediction Report

    Predicted Condition: {predicted_label}

    Symptoms Provided:
    {text}

    Suggested Recommendations:
    - Consult an endocrinologist.
    - Take proper thyroid function tests (TSH, T3, T4).
    - Maintain a balanced iodine-rich diet.
    - Regular medical follow-up is recommended.
    - Avoid stress and maintain healthy lifestyle.

    This is an AI-assisted prediction. Please consult a medical professional.
    """


# ─────────────────────────────────────────────
# User Management
# ─────────────────────────────────────────────

def load_users():
    if os.path.exists('users.json'):
        with open('users.json', 'r') as file:
            return json.load(file)
    return {}

def save_users(users):
    with open('users.json', 'w') as file:
        json.dump(users, file, indent=4)

def add_user(username, password, full_name, email, phone, dob, gender, address, medical_history):
    users = load_users()
    if username in users:
        return False
    users[username] = {
        "password": password,
        "full_name": full_name,
        "email": email,
        "phone": phone,
        "dob": dob,
        "gender": gender,
        "address": address,
        "medical_history": medical_history,
        "history": [],
        "consultations": []
    }
    save_users(users)
    return True

def validate_user(username, password):
    users = load_users()
    user = users.get(username)
    if user and isinstance(user, dict):
        return user.get("password") == password
    return False


# ─────────────────────────────────────────────
# Existing Routes
# ─────────────────────────────────────────────

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if validate_user(username, password):
            session['user'] = username
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        full_name = request.form['full_name']
        email = request.form['email']
        phone = request.form['phone']
        dob = request.form['dob']
        gender = request.form['gender']
        address = request.form['address']
        medical_history = request.form['medical_history']

        if add_user(username, password, full_name, email, phone, dob, gender, address, medical_history):
            flash('Registration successful. Please log in.')
            return redirect(url_for('login'))
        else:
            flash('Username already exists')
    return render_template('register.html')

@app.route('/index')
def index():
    if 'user' in session:
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('landing'))

@app.route('/result', methods=['POST'])
def result():
    if 'user' not in session:
        return redirect(url_for('login'))

    text = request.form['text']
    image = request.files['image']
    image_path = f"static/uploads/{image.filename}"
    image.save(image_path)

    predicted_label_text, text_input = inference_text(model, tokenizer, text, device)
    report_text = generate_report(predicted_label_text, text_input, "")

    predicted_label_image, image_input = inference_image(model, image_path, image_transforms, device)
    report_image = generate_report(predicted_label_image, "", image_input)

    predicted_label_both, text_input, image_input = inference_both(model, tokenizer, text, image_path, image_transforms, device)
    report_both = generate_report(predicted_label_both, text_input, image_input)

    # ── Save to Patient History ──
    users = load_users()
    username = session['user']
    if username in users:
        if 'history' not in users[username]:
            users[username]['history'] = []
        users[username]['history'].append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symptoms": text,
            "prediction_text": predicted_label_text,
            "prediction_image": predicted_label_image,
            "prediction_both": predicted_label_both
        })
        save_users(users)

    return render_template('result.html',
                           predicted_label_text=predicted_label_text,
                           predicted_label_image=predicted_label_image,
                           predicted_label_both=predicted_label_both,
                           report_text=report_text,
                           report_image=report_image,
                           report_both=report_both)


# ─────────────────────────────────────────────
# NEW ROUTE 1: Chatbot / AI Assistant
# ─────────────────────────────────────────────

@app.route('/chatbot')
def chatbot():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('chatbot.html')

@app.route('/chatbot/ask', methods=['POST'])
def chatbot_ask():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'reply': 'Please enter a message.'})

    try:
        # ── HuggingFace Free API ──
        HF_API_KEY = "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"   # <-- PASTE YOUR TOKEN HERE
        API_URL = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": "Bearer " + HF_API_KEY,
            "Content-Type": "application/json"
        }

        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct:cerebras",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are ThyroidCare AI, a medical assistant specialized in thyroid diseases. "
                        "Give detailed, clear answers with at least 3-5 points. "
                        "Use simple language. Always end with: Please consult your doctor for personalized advice."
                    )
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }

        import time
        reply = None
        for attempt in range(3):
            response = req.post(API_URL, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                reply = result["choices"][0]["message"]["content"].strip()
                break
            elif response.status_code == 503:
                time.sleep(10)
                continue
            else:
                reply = "Error " + str(response.status_code) + ": " + response.text[:300]
                break

        if not reply:
            reply = "Sorry, the model is busy. Please try again in 30 seconds."

    except Exception as e:
        reply = "Error: " + str(e)

    return jsonify({'reply': reply})


# ─────────────────────────────────────────────
# NEW ROUTE 2: Patient History
# ─────────────────────────────────────────────

@app.route('/history')
def patient_history():
    if 'user' not in session:
        return redirect(url_for('login'))

    users = load_users()
    username = session['user']
    user_data = users.get(username, {})
    history = user_data.get('history', [])
    history = list(reversed(history))

    return render_template('history.html', history=history, username=username)

@app.route('/history/delete/<int:index>', methods=['POST'])
def delete_history(index):
    if 'user' not in session:
        return redirect(url_for('login'))

    users = load_users()
    username = session['user']
    if username in users and 'history' in users[username]:
        history = users[username]['history']
        real_index = len(history) - 1 - index
        if 0 <= real_index < len(history):
            history.pop(real_index)
            users[username]['history'] = history
            save_users(users)
            flash('Record deleted successfully.')
    return redirect(url_for('patient_history'))


# ─────────────────────────────────────────────
# NEW ROUTE 3: Doctor Consultation
# ─────────────────────────────────────────────

DOCTORS = [
    {"id": 1, "name": "Dr. Aisha Sharma", "specialty": "Endocrinologist",   "hospital": "Apollo Hospitals", "city": "Chennai", "available": "Mon, Wed, Fri — 10AM to 2PM", "contact": "aisha.sharma@apollo.com"},
    {"id": 2, "name": "Dr. Rajesh Kumar",  "specialty": "Thyroid Specialist", "hospital": "Fortis Hospital",  "city": "Chennai", "available": "Tue, Thu — 9AM to 1PM",      "contact": "rajesh.kumar@fortis.com"},
    {"id": 3, "name": "Dr. Priya Nair",    "specialty": "Endocrinologist",   "hospital": "MIOT Hospital",    "city": "Chennai", "available": "Mon to Sat — 3PM to 6PM",    "contact": "priya.nair@miot.com"},
    {"id": 4, "name": "Dr. Suresh Babu",   "specialty": "General Physician",  "hospital": "Kauvery Hospital", "city": "Chennai", "available": "Daily — 8AM to 12PM",         "contact": "suresh.babu@kauvery.com"},
]

@app.route('/consultation')
def consultation():
    if 'user' not in session:
        return redirect(url_for('login'))

    users = load_users()
    username = session['user']
    user_data = users.get(username, {})
    my_consultations = user_data.get('consultations', [])

    return render_template('consultation.html', doctors=DOCTORS, my_consultations=my_consultations)

@app.route('/consultation/book', methods=['POST'])
def book_consultation():
    if 'user' not in session:
        return redirect(url_for('login'))

    doctor_id = int(request.form['doctor_id'])
    preferred_date = request.form['preferred_date']
    preferred_time = request.form['preferred_time']
    notes = request.form.get('notes', '')

    doctor = next((d for d in DOCTORS if d['id'] == doctor_id), None)
    if not doctor:
        flash('Doctor not found.')
        return redirect(url_for('consultation'))

    users = load_users()
    username = session['user']
    if username in users:
        if 'consultations' not in users[username]:
            users[username]['consultations'] = []
        users[username]['consultations'].append({
            "booking_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "doctor_name": doctor['name'],
            "specialty": doctor['specialty'],
            "hospital": doctor['hospital'],
            "preferred_date": preferred_date,
            "preferred_time": preferred_time,
            "notes": notes,
            "status": "Pending"
        })
        save_users(users)
        flash(f'Appointment booked successfully with {doctor["name"]}!')

    return redirect(url_for('consultation'))


# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
