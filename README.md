AI Disease Diagnosis Chatbot
A Flask-based AI application that predicts diseases using symptoms entered by the user.
The system uses two deep learning models (LSTM and CNN) trained on a symptom–disease dataset and automatically selects the best model based on prediction confidence.
The app also displays:
 Detailed disease description
 Precautions
 Confidence level
Comparison of LSTM vs CNN predictions
A clean modern UI with a medical background


Features
✔ Predicts disease from entered symptoms
✔ Uses TF-IDF + LSTM and TF-IDF + CNN
✔ Automatically selects the most confident model
✔ Shows:
 Best model result
 Description
 Precautions
 Confidence score
Comparison of both model scores
✔ Clean UI with background image
✔ Easy to run locally
✔ Completely offline — no API required
✔ Both models trained inside a separate training folder

📁 Project Structure

AI_Disease_Diagnosis/
│
├── app.py
├── requirements.txt
│
├── models/
│   ├── disease_lstm_model.h5
│   ├── disease_cnn_model.h5
│   ├── label_encoder.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│
├── templates/
│   ├── index.html
│   ├── result.html
│
└── static/
    ├── background.png

Training Folder
AI_Disease_Diagnosis_Training/
│
├── train.py
├── dataset.csv
├── Symptom-severity.csv
├── symptom_Description.csv
├── symptom_precaution.csv
└── models/
You train the ML models here and copy the outputs (.h5, .pkl) into the Flask project's models/ folder.

Technologies Used

| Component       | Technology                    |
| --------------- | ----------------------------- |
| Backend         | Flask                         |
| Models          | TensorFlow/Keras (LSTM & CNN) |
| Preprocessing   | TF-IDF Vectorizer             |
| Language        | Python                        |
| UI              | HTML + Inline CSS             |
| Storage         | Local Models & CSV files      |
| Version Control | Git + GitHub                  |

Installation & Run Instructions
1️⃣ Clone the Repository
git clone https://github.com/saikaushik7/AI_Disease_Diagnosis.git
cd AI_Disease_Diagnosis

2️⃣ Create & Activate Environment
conda create -n disease python=3.10 -y
conda activate disease

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Flask Application
python app.py

Model Overview

🔵 LSTM Model
Input: TF-IDF Vector
Layers: LSTM → Dropout → Dense → Softmax
Captures sequence-like structure of symptoms

🔴 CNN Model
Input: TF-IDF Vector reshaped to 1D
Layers: Conv1D → MaxPool → Conv1D → GlobalMaxPool → Dense
Extracts pattern filters from symptom vectors

🟢 Best Model Selection Logic
The app picks the model with the highest softmax confidence score.

Dataset
This project uses CSV files:
 dataset.csv
 Symptom-severity.csv
 symptom_Description.csv
 symptom_precaution.csv

These files provide:
 Symptoms
 Disease labels
 Severity levels
 Descriptions
 Precautions



 Sai Kaushik
GitHub: saikaushik7
