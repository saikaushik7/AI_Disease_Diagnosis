from flask import Flask, render_template, request
import numpy as np
import pickle
import os
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# -------------------------------
# Load models & encoders
# -------------------------------
lstm_model = load_model(os.path.join(MODEL_DIR, "disease_lstm_model.h5"))
cnn_model  = load_model(os.path.join(MODEL_DIR, "disease_cnn_model.h5"))

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

# -------------------------------
# Load descriptions & precautions (if available)
# -------------------------------
# Expected CSV format:
# symptom_Description.csv -> columns: Disease, Description
# symptom_precaution.csv  -> columns: Disease, precaution_1, precaution_2, ...
DESC_DF = None
PREC_DF = None
desc_path = os.path.join(MODEL_DIR, "symptom_Description.csv")
prec_path = os.path.join(MODEL_DIR, "symptom_precaution.csv")

if os.path.exists(desc_path):
    try:
        DESC_DF = pd.read_csv(desc_path)
    except Exception:
        DESC_DF = None

if os.path.exists(prec_path):
    try:
        PREC_DF = pd.read_csv(prec_path)
    except Exception:
        PREC_DF = None

def get_description(disease_name):
    if DESC_DF is None:
        return "No description available."
    row = DESC_DF[DESC_DF['Disease'] == disease_name]
    if row.empty:
        return "No description available."
    # assume column 'Description'
    return row.iloc[0].get('Description', "No description available.")

def get_precautions(disease_name):
    if PREC_DF is None:
        return []
    row = PREC_DF[PREC_DF['Disease'] == disease_name]
    if row.empty:
        return []
    # all other columns are precautions
    vals = row.iloc[0].tolist()[1:]  # drop Disease
    # filter nan / empty
    return [str(v) for v in vals if pd.notna(v) and str(v).strip() != ""]

# -------------------------------
# Prediction logic: returns (best_model_name, disease_name, score)
# -------------------------------
def predict_disease_best(user_input):
    # Normalize and tokenize input (same preprocessing you used during training)
    symptoms_list = user_input.lower().replace(",", " ").split()
    symptom_text = " ".join(symptoms_list)

    # Vectorize
    symptom_vec = vectorizer.transform([symptom_text]).toarray()

    # Prepare inputs
    lstm_input = symptom_vec.reshape(1, 1, symptom_vec.shape[1])
    cnn_input  = symptom_vec.reshape(1, symptom_vec.shape[1], 1)

    # Predict probabilities
    lstm_probs = lstm_model.predict(lstm_input)  # shape (1, num_classes)
    cnn_probs  = cnn_model.predict(cnn_input)

    lstm_confidence = float(np.max(lstm_probs))
    cnn_confidence  = float(np.max(cnn_probs))

    lstm_pred_idx = int(np.argmax(lstm_probs, axis=1)[0])
    cnn_pred_idx  = int(np.argmax(cnn_probs, axis=1)[0])

    lstm_disease = le.inverse_transform([lstm_pred_idx])[0]
    cnn_disease  = le.inverse_transform([cnn_pred_idx])[0]

    # Choose the model with higher confidence
    if lstm_confidence >= cnn_confidence:
        return {
            "model": "LSTM",
            "disease": lstm_disease,
            "confidence": round(lstm_confidence, 4),
            "all": {
                "LSTM": {"disease": lstm_disease, "confidence": round(lstm_confidence, 4)},
                "CNN":  {"disease": cnn_disease,  "confidence": round(cnn_confidence, 4)}
            }
        }
    else:
        return {
            "model": "CNN",
            "disease": cnn_disease,
            "confidence": round(cnn_confidence, 4),
            "all": {
                "LSTM": {"disease": lstm_disease, "confidence": round(lstm_confidence, 4)},
                "CNN":  {"disease": cnn_disease,  "confidence": round(cnn_confidence, 4)}
            }
        }

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    user_symptoms = request.form.get("symptoms", "").strip()
    if user_symptoms == "":
        return render_template("index.html", error="Please enter symptoms.")

    result = predict_disease_best(user_symptoms)
    disease = result["disease"]
    model_used = result["model"]
    confidence = result["confidence"]

    description = get_description(disease)
    precautions = get_precautions(disease)

    # Render result.html with single best result + description + precautions
    return render_template(
        "result.html",
        symptoms=user_symptoms,
        model_used=model_used,
        disease=disease,
        confidence=confidence,
        description=description,
        precautions=precautions,
        all_results=result["all"]
    )


if __name__ == "__main__":
    app.run(debug=True)
