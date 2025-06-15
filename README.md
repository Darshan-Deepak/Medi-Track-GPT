# 🏥 MediTrack-GPT: Discharge Summary Triage Risk Predictor

MediTrack-GPT is an intelligent triage assistant designed to assess patient risk levels from discharge summaries. This project fine-tunes the BioBERT model to classify clinical notes into risk categories (Low, Moderate, High) and delivers real-time predictions via a clean Streamlit UI.

---

## 📌 Project Overview

**Goal**: Automatically assess patient triage risk (Low, Moderate, High) using their discharge summary.

**Input**: Free-text discharge summary from EHR records  
**Output**: Predicted Triage Risk + Confidence Score

---

## 🧠 Model Details

- ✅ **Model Base**: `biobert-base-cased-v1.1` (from dmis-lab)
- 🔬 **Fine-tuned on**: Synthetic Synthea EHR data (discharge summaries)
- 🔍 **Task**: Sequence classification with 3 labels (Low, Moderate, High)
- 📈 **Evaluation**: Trained for 3 epochs with weight decay, evaluated per epoch

---

## 🗂️ Folder Structure

```
MediTrack-GPT/
├── data/
│   └── ehr/                         # Synthetic patient data from Synthea
├── notebooks/
│   └── discharge_summary_generator.ipynb  # Fine-tuning & prediction pipeline
├── results/                         # Prediction CSVs and logs
├── triage-biobert-model/            # Fine-tuned model and tokenizer files
├── triage_predictor_ui.py          # Streamlit app for triage prediction
├── README.md                        # You're here!
```

---

## 🚀 How to Use

### 1. Fine-tune (Optional)

You can run `notebooks/discharge_summary_generator.ipynb` to retrain or test the model pipeline. This includes:

- Data loading and summary generation
- Model fine-tuning (BioBERT)
- Label + confidence predictions
- Visualizations (Seaborn)

### 2. Launch Streamlit UI

```
streamlit run triage_predictor_ui.py
```

You’ll get a web UI where you can paste any discharge summary to get:
- 🧾 Predicted risk label
- 📊 Model confidence

---

## 📊 Visualizations

- Triage risk label distribution (countplot)
- Confidence score histogram by class
- Top highest confidence risk cases (preview)

---

## ✅ Requirements

- Python 3.10+
- `transformers==4.52.4`
- `torch`, `pandas`, `seaborn`, `matplotlib`
- `streamlit`

Install with:

```bash
pip install -r requirements.txt
```

---

## 🙌 Credits

- BioBERT by DMIS Lab ([Hugging Face](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1))
- Synthetic data by [Synthea](https://synthetichealth.github.io/synthea/)
- Project design and implementation by **Darshan**

---

## 📌 Future Scope

- Integrate patient demographics into model features
- Add XAI support (e.g., LIME or SHAP for interpretability)
- Deploy on Hugging Face Spaces or Streamlit Cloud

---

## 💡 Example

```text
Encounter ID: 7b2e8293
Reason: Chest pain, shortness of breath
Diagnosis: Possible myocardial infarction
Vitals: Elevated heart rate, low oxygen saturation
Plan: Discharge with aspirin & statins, return if worsens

➡️ Prediction: High Risk (Confidence: 0.99)
```
