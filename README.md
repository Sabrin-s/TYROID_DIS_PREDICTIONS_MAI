# 🩺 Attention-Based Multi-Modal Framework for Thyroid Disease Diagnosis and Prediction
AI-based Thyroid Disease Prediction and Management System developed to help in the early detection and monitoring of thyroid disorders. The system uses machine learning and deep learning techniques to analyze both patient symptoms and thyroid scan images for accurate disease prediction. It combines text analysis using BART and image analysis using ResNet18 in a multimodal model to identify thyroid-related conditions. The application is developed using Python and Flask with HTML and CSS for the frontend. It also includes features such as AI chatbot support, patient history management, report generation, and doctor consultation booking. The main objective of the system is to provide a fast, intelligent, and user-friendly healthcare solution that improves diagnosis support and patient care.
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-green)
![Deep Learning](https://img.shields.io/badge/AI-Deep%20Learning-red)
![IEEE](https://img.shields.io/badge/Published-IEEE%20ICCNCT--2026-orange)

## 📌 About the Project

An AI-powered web application that diagnoses thyroid diseases 
(Hypothyroidism, Hyperthyroidism, Thyroid Cancer, Normal) 
by combining:
- 🔤 **Clinical Text Analysis** using BERT
- 🖼️ **Ultrasound Image Analysis** using ResNet-18
- 🔗 **Multi-Modal Fusion** for combined prediction

> Published at IEEE ICCNCT-2026 Conference  
> Mahendra Engineering College (Autonomous), Namakkal

---

## 👩‍💻 Team Members

| Name | Register No |
|------|------------|
| Sabrin S | 6113221031129 |
| Sandhiya S | 6113221031131 |
| Sowmiya B | 6113221031154 |
| Padma Rekha T | 6113221033810 |

**Guide:** Ms. N. Karthigavani, M.E., Assistant Professor, CSE

---

## ✨ Features

- 🔐 User Registration & Login
- 🩻 Upload Thyroid Ultrasound Images
- 📝 Enter Patient Symptoms (Text)
- 🤖 AI Diagnosis (Text + Image + Combined)
- 📊 Diagnosis History
- 💬 AI Thyroid Chatbot Assistant
- 👨‍⚕️ Doctor Consultation Booking
- 📄 Print Diagnosis Report

---

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 85.7% | 84.2% | 83.5% | 83.8% |
| CNN | 90.2% | 89.5% | 90.0% | 89.7% |
| DNN | 88.3% | 87.1% | 86.5% | 86.8% |
| **DMMSCN (Ours)** | **96.5%** | **95.8%** | **96.0%** | **95.9%** |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, TailwindCSS |
| Backend | Python, Flask |
| ML Models | XGBoost, ResNet-18, BERT |
| Deep Learning | PyTorch |
| Database | JSON-based |

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Sabrin-s/TYROID_DIS_PREDICTIONS_MAI.git
cd TYROID_DIS_PREDICTIONS_MAI
```

### 2. Install dependencies
```bash
cd WEB_APP
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
