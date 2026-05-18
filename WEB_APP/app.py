from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import json
import os
import time
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from transformers import BartModel, BartTokenizer
from torchvision import models
import pandas as pd
import requests as req
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────
# Flask App Configuration
# ─────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = "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"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────
# Hugging Face API Key
# ─────────────────────────────────────────────

HF_API_KEY = ""

# ─────────────────────────────────────────────
# Doctors List
# ─────────────────────────────────────────────

DOCTORS = [
    {
        "id": 1,
        "name": "Dr. Anitha Ramesh",
        "specialty": "Endocrinologist",
        "hospital": "Apollo Hospitals, Chennai",
        "experience": "15 years",
        "rating": 4.8,
        "image": "doctor1.jpg"
    },
    {
        "id": 2,
        "name": "Dr. Karthik Selvam",
        "specialty": "Thyroid Specialist",
        "hospital": "Fortis Malar Hospital, Chennai",
        "experience": "12 years",
        "rating": 4.7,
        "image": "doctor2.jpg"
    },
    {
        "id": 3,
        "name": "Dr. Priya Nair",
        "specialty": "Endocrinologist",
        "hospital": "MIOT International, Chennai",
        "experience": "10 years",
        "rating": 4.6,
        "image": "doctor3.jpg"
    },
    {
        "id": 4,
        "name": "Dr. Suresh Kumar",
        "specialty": "Internal Medicine",
        "hospital": "Kauvery Hospital, Trichy",
        "experience": "18 years",
        "rating": 4.9,
        "image": "doctor4.jpg"
    },
    {
        "id": 5,
        "name": "Dr. Meena Sundaram",
        "specialty": "Thyroid Specialist",
        "hospital": "Sri Ramachandra Hospital, Chennai",
        "experience": "8 years",
        "rating": 4.5,
        "image": "doctor5.jpg"
    },
]

# ─────────────────────────────────────────────
# Load Tokenizer
# ─────────────────────────────────────────────

model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)

# ─────────────────────────────────────────────
# Image Transformations
# ─────────────────────────────────────────────

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ─────────────────────────────────────────────
# Load Dataset
# ─────────────────────────────────────────────

text_df = pd.read_csv("data.csv")
label_list = text_df['label'].unique().tolist()

# ─────────────────────────────────────────────
# Device Configuration
# ─────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "multimodal_model.pth"

# ─────────────────────────────────────────────
# Load ResNet18
# ─────────────────────────────────────────────

image_model = models.resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1
)

image_model.fc = nn.Identity()

# ─────────────────────────────────────────────
# Multimodal Classifier
# ─────────────────────────────────────────────

class MultiModalClassifier(nn.Module):

    def __init__(
        self,
        text_model,
        image_model,
        text_feat_dim,
        image_feat_dim,
        hidden_dim,
        num_classes
    ):

        super(MultiModalClassifier, self).__init__()

        self.text_model = text_model
        self.image_model = image_model

        self.text_fc = nn.Linear(text_feat_dim, hidden_dim)
        self.image_fc = nn.Linear(image_feat_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text_input=None, image_input=None):

        features = None

        # Text Features
        if text_input is not None:

            text_input_filtered = {
                k: v for k, v in text_input.items()
                if k != "labels"
            }

            text_outputs = self.text_model(**text_input_filtered)

            pooled_text = text_outputs.last_hidden_state.mean(dim=1)

            text_features = self.text_fc(pooled_text)

            features = (
                text_features
                if features is None
                else features + text_features
            )

        # Image Features
        if image_input is not None:

            image_features = self.image_model(image_input)

            image_features = self.image_fc(image_features)

            features = (
                image_features
                if features is None
                else features + image_features
            )

        # Average if both exist
        if (text_input is not None) and (image_input is not None):
            features = features / 2

        logits = self.classifier(features)

        return logits

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────

model = MultiModalClassifier(
    text_model=BartModel.from_pretrained(model_name),
    image_model=image_model,
    text_feat_dim=768,
    image_feat_dim=512,
    hidden_dim=512,
    num_classes=len(label_list)
)

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model.to(device)
model.eval()

# ─────────────────────────────────────────────
# Label Mapping
# ─────────────────────────────────────────────

label2id = {
    label: idx
    for idx, label in enumerate(label_list)
}

id2label = {
    idx: label
    for label, idx in label2id.items()
}

# ─────────────────────────────────────────────
# Inference Functions
# ─────────────────────────────────────────────

def inference_text(model, tokenizer, text, device, max_length=128):

    model.eval()

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    for key in encoding:
        encoding[key] = encoding[key].to(device)

    with torch.no_grad():

        logits = model(
            text_input=encoding,
            image_input=None
        )

    probabilities = torch.softmax(logits, dim=1)

    pred_id = torch.argmax(probabilities, dim=1).item()

    confidence = probabilities[0][pred_id].item()

    return id2label[pred_id], confidence, text


def inference_image(model, image_path, transform, device):

    model.eval()

    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(
            text_input=None,
            image_input=image
        )

    probabilities = torch.softmax(logits, dim=1)

    pred_id = torch.argmax(probabilities, dim=1).item()

    confidence = probabilities[0][pred_id].item()

    return id2label[pred_id], confidence, image_path


def inference_both(
    model,
    tokenizer,
    text,
    image_path,
    transform,
    device,
    max_length=128
):

    model.eval()

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    for key in encoding:
        encoding[key] = encoding[key].to(device)

    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(
            text_input=encoding,
            image_input=image
        )

    probabilities = torch.softmax(logits, dim=1)

    pred_id = torch.argmax(probabilities, dim=1).item()

    confidence = probabilities[0][pred_id].item()

    return id2label[pred_id], confidence, text, image_path

# ─────────────────────────────────────────────
# Generate Report
# ─────────────────────────────────────────────

def generate_report(predicted_label, confidence, text):

    return f"""
    Thyroid Disease Prediction Report

    Predicted Condition:
    {predicted_label}

    Confidence:
    {confidence:.2f}

    Symptoms:
    {text}

    Recommendations:
    - Consult endocrinologist
    - Take TSH, T3, T4 tests
    - Reduce stress
    - Eat healthy food
    - Maintain proper sleep

    AI-assisted prediction only.
    """

# ─────────────────────────────────────────────
# User Management
# ─────────────────────────────────────────────

def load_users():

    if os.path.exists("users.json"):

        with open("users.json", "r") as file:
            return json.load(file)

    return {}


def save_users(users):

    with open("users.json", "w") as file:
        json.dump(users, file, indent=4)


def add_user(
    username,
    password,
    full_name,
    email,
    phone,
    dob,
    gender,
    address,
    medical_history
):

    users = load_users()

    username = username.strip().lower()

    if username in users:
        return False

    users[username] = {

        "password": generate_password_hash(password),

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

    username = username.strip().lower()

    user = users.get(username)

    if not user:
        return False

    saved_password = user.get("password", "")

    try:
        return check_password_hash(saved_password, password)
    except:
        return False

# ─────────────────────────────────────────────
# Landing Page
# ─────────────────────────────────────────────

@app.route("/")
def landing():
    return render_template("landing.html")

# ─────────────────────────────────────────────
# Login
# ─────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"].strip().lower()
        password = request.form["password"]

        if validate_user(username, password):

            session["user"] = username

            flash("Login successful")

            return redirect(url_for("index"))

        else:
            flash("Invalid username or password")

    return render_template("login.html")

# ─────────────────────────────────────────────
# Register
# ─────────────────────────────────────────────

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"].strip().lower()

        password = request.form["password"]

        full_name = request.form["full_name"]
        email = request.form["email"]
        phone = request.form["phone"]
        dob = request.form["dob"]
        gender = request.form["gender"]
        address = request.form["address"]
        medical_history = request.form["medical_history"]

        success = add_user(
            username,
            password,
            full_name,
            email,
            phone,
            dob,
            gender,
            address,
            medical_history
        )

        if success:

            flash("Registration successful")

            return redirect(url_for("login"))

        else:
            flash("Username already exists")

    return render_template("register.html")

# ─────────────────────────────────────────────
# Home
# ─────────────────────────────────────────────

@app.route("/index")
def index():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("index.html")

# ─────────────────────────────────────────────
# Logout
# ─────────────────────────────────────────────

@app.route("/logout")
def logout():

    session.pop("user", None)

    return redirect(url_for("landing"))

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────

@app.route("/result", methods=["POST"])
def result():

    if "user" not in session:
        return redirect(url_for("login"))

    text = request.form["text"]

    image = request.files["image"]

    filename = secure_filename(image.filename)

    image_path = os.path.join(UPLOAD_FOLDER, filename)

    image.save(image_path)

    predicted_label_text, confidence_text, text_input = inference_text(
        model,
        tokenizer,
        text,
        device
    )

    predicted_label_image, confidence_image, image_input = inference_image(
        model,
        image_path,
        image_transforms,
        device
    )

    predicted_label_both, confidence_both, text_input, image_input = inference_both(
        model,
        tokenizer,
        text,
        image_path,
        image_transforms,
        device
    )

    report_text = generate_report(
        predicted_label_both,
        confidence_both,
        text
    )

    users = load_users()

    username = session["user"]

    if username in users:

        users[username]["history"].append({

            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            "symptoms": text,

            "prediction_text": predicted_label_text,

            "prediction_image": predicted_label_image,

            "prediction_both": predicted_label_both
        })

        save_users(users)

    return render_template(
        "result.html",

        predicted_label_text=predicted_label_text,
        predicted_label_image=predicted_label_image,
        predicted_label_both=predicted_label_both,

        confidence_text=confidence_text,
        confidence_image=confidence_image,
        confidence_both=confidence_both,

        report_text=report_text
    )

# ─────────────────────────────────────────────
# Chatbot
# ─────────────────────────────────────────────

@app.route("/chatbot")
def chatbot():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("chatbot.html")


@app.route("/chatbot/ask", methods=["POST"])
def chatbot_ask():

    if "user" not in session:
        return jsonify({"reply": "Unauthorized"})

    data = request.get_json()

    user_message = data.get("message")

    if not user_message:
        return jsonify({"reply": "Empty message"})

    msg = user_message.lower()

    if "hypothyroidism" in msg:
        reply = "Hypothyroidism occurs when thyroid hormone levels are low. Symptoms include fatigue, weight gain, and cold sensitivity."

    elif "hyperthyroidism" in msg:
        reply = "Hyperthyroidism means overactive thyroid gland. Symptoms include weight loss, anxiety, and rapid heartbeat."

    elif "tsh" in msg:
        reply = "TSH test measures thyroid stimulating hormone and helps evaluate thyroid function."

    elif "diet" in msg:
        reply = "Healthy thyroid diet includes iodine, selenium, eggs, fish, nuts, and vegetables."

    else:
        reply = "Please consult a doctor for accurate medical advice."

    return jsonify({"reply": reply})

# ─────────────────────────────────────────────
# History
# ─────────────────────────────────────────────

@app.route("/history")
def patient_history():

    if "user" not in session:
        return redirect(url_for("login"))

    users = load_users()
    username = session["user"]
    history = users.get(username, {}).get("history", [])

    # Reverse so newest appears first
    history_reversed = list(reversed(history))

    return render_template(
        "history.html",
        history=history_reversed
    )


@app.route("/delete_history/<int:index>", methods=["POST"])
def delete_history(index):

    if "user" not in session:
        return redirect(url_for("login"))

    users = load_users()
    username = session["user"]

    if username in users:
        history = users[username].get("history", [])
        # history.html shows records reversed, so convert index back to actual
        actual_index = len(history) - 1 - index
        if 0 <= actual_index < len(history):
            del history[actual_index]
            users[username]["history"] = history
            save_users(users)
            flash("Record deleted successfully")

    return redirect(url_for("patient_history"))

# ─────────────────────────────────────────────
# Consultation
# ─────────────────────────────────────────────

@app.route("/consultation")
def consultation():

    if "user" not in session:
        return redirect(url_for("login"))

    users = load_users()
    username = session["user"]
    my_consultations = users.get(username, {}).get("consultations", [])

    return render_template(
        "consultation.html",
        doctors=DOCTORS,
        my_consultations=my_consultations
    )


@app.route("/book_consultation", methods=["POST"])
def book_consultation():

    if "user" not in session:
        return redirect(url_for("login"))

    doctor_id = int(request.form["doctor_id"])
    preferred_date = request.form["preferred_date"]
    preferred_time = request.form["preferred_time"]
    notes = request.form.get("notes", "")

    doctor = next((d for d in DOCTORS if d["id"] == doctor_id), None)

    if not doctor:
        flash("Doctor not found")
        return redirect(url_for("consultation"))

    users = load_users()
    username = session["user"]

    if username in users:
        users[username].setdefault("consultations", []).append({
            "doctor_name": doctor["name"],
            "specialty": doctor["specialty"],
            "hospital": doctor["hospital"],
            "preferred_date": preferred_date,
            "preferred_time": preferred_time,
            "notes": notes,
            "status": "Pending",
            "booking_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        save_users(users)

    flash("Appointment booked successfully!")
    return redirect(url_for("consultation"))

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )
