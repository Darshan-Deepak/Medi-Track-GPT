# triage_ui.py

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model and tokenizer
MODEL_PATH = "./triage-biobert-model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

label_map = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk"
}

def predict(summary_text):
    inputs = tokenizer(summary_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs).item()
    return label_map[pred], round(probs[0][pred].item(), 4)

# Streamlit UI
st.set_page_config(page_title="Triage Predictor", layout="centered")
st.title("🏥 MediTrack-GPT: Triage Risk Predictor")
st.markdown("Enter a patient's **discharge summary** below to predict the triage risk level.")

summary_input = st.text_area("Discharge Summary", height=250)

if st.button("Predict Triage Risk"):
    if summary_input.strip():
        label, confidence = predict(summary_input)
        st.success(f"**Triage Risk:** {label}\n\n**Confidence:** {confidence:.2f}")
    else:
        st.warning("Please enter a discharge summary.")
