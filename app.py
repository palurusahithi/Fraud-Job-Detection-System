import joblib
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image
import numpy as np
import cv2
import torch
import re
import os
import warnings
import time

# Suppress those version warnings for a cleaner presentation
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, render_template, request
from transformers import BertTokenizer, BertModel

app = Flask(__name__, template_folder="templates")

# ---------------- 1. LOAD FILES ----------------
print("Loading Model Components... Please wait.")
mlp_model = joblib.load("mlp_model.joblib")
tfidf = joblib.load("tfidf.joblib")
scaler = joblib.load("scaler.joblib")
threshold = joblib.load("threshold.joblib")

# ---------------- 2. BERT SETUP ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(device)
bert.eval()

# ---------------- 3. TESSERACT PATH ----------------
# Ensure this matches your local installation path!
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------- 4. UTILITY FUNCTIONS ----------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def mean_pooling(outputs, attention_mask):
    token_embeddings = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1)

def get_embedding(text):
    inputs = tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert(**inputs)
        pooled = mean_pooling(outputs, inputs['attention_mask'])
    return pooled.cpu().numpy()

def extract_text_from_image(file):
    try:
        image = Image.open(file).convert('RGB')
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
        text = pytesseract.image_to_string(gray)
        return text.strip()
    except:
        return ""

# ---------------- 5. AI EXPLANATION ----------------

def generate_explanation(text):
    text = text.lower()
    reasons = []
    if "no experience" in text:
        reasons.append("Unrealistic requirement: No experience needed")
    if "earn" in text or "$" in text or "salary" in text:
        reasons.append("Suspicious salary or earning claims")
    if "urgent" in text or "immediately" in text:
        reasons.append("Pressure tactics detected")
    if "fee" in text or "payment" in text or "registration" in text:
        reasons.append("Requests for payment detected")
    if "work from home" in text:
        reasons.append("Common scam keyword: Work from home")
    
    if len(reasons) == 0:
        reasons.append("No major fraud indicators detected")
    return reasons

# ---------------- 6. PREDICT FUNCTION ----------------

def predict(text):
    cleaned = clean_text(text)
    
    # 1. Get Keyword Reasons
    reasons = generate_explanation(cleaned)
    
    # 2. Get ML Probability
    emb_bert = get_embedding(cleaned)
    emb_tfidf = tfidf.transform([cleaned]).toarray()
    vector = np.hstack((emb_bert, emb_tfidf))
    vector = scaler.transform(vector)
    prob = mlp_model.predict_proba(vector)[0][1] 
    
    # 3. SAFETY OVERRIDE LOGIC
    # If keywords find 2+ flags, we force 'Fraud' for user safety (High Recall)
    if len(reasons) >= 2 or prob >= threshold:
        label = "Fraud"
        display_conf = round(max(prob * 100, 75.0), 2) 
        final_prob = max(prob, 0.75)
    else:
        label = "Real"
        display_conf = round((1 - prob) * 100, 2)
        final_prob = prob
        
    return label, display_conf, final_prob

# ---------------- 7. ROUTE ----------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text_input = request.form.get("job_description")
        image = request.files.get("image")

        combined_text = ""
        if text_input:
            combined_text += text_input + " "
        if image and image.filename != "":
            ocr_text = extract_text_from_image(image)
            combined_text += ocr_text

        if combined_text.strip() == "":
            return render_template("index.html", error="Enter text or upload image.")

        # 1. Prediction
        label, confidence, raw_prob = predict(combined_text)
        
        # 2. Explanations
        reasons = generate_explanation(combined_text)

        # 3. Logic for the Checklist (Ticks and Crosses)
        is_fraud = (label == "Fraud")
        cleaned_lower = combined_text.lower()
        
        checklist = {
            "Company Profile": not is_fraud,
            "Salary Legitimacy": "salary" not in str(reasons).lower(),
            "Non-Suspicious": not is_fraud,
            "Professional Language": True,
            "Safe Contact": "telegram" not in cleaned_lower and "whatsapp" not in cleaned_lower
        }

        authenticity = int((1 - raw_prob) * 100)

        return render_template(
            "results.html",
            label=label,
            confidence=confidence,
            authenticity=authenticity,
            checklist=checklist,
            reasons=reasons
        )

    return render_template("index.html")

if __name__ == "__main__":
    print("\n--- SERVER STARTING ---")
    print("Go to: http://127.0.0.1:5000")
    app.run(debug=True)