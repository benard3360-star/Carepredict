# app.py
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load Model, Top Features, Encoder, and Scaler
with open("catboost_tuned_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("top_features.pkl", "rb") as f:
    top_features = pickle.load(f)

with open("categorical_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("scaler_stepwise_features.pkl", "rb") as f:
    scaler = pickle.load(f)

# Form categorical features
form_categorical_features = ['Final Grade', 'County', 'Sub_County', 'Location', 'Course']

# Correct encoder mapping based on actual encoder contents
cat_to_encoder_idx = {
    'County': 2,
    'Sub_County': 3,
    'Location': 4,
    'Course': 7,
    'Final Grade': 9
}

categorical_values = {feat: encoders[idx].classes_.tolist() for feat, idx in cat_to_encoder_idx.items()}

# Create ordered features list with type information
features_info = [
    {'name': feat, 'type': 'categorical' if feat in form_categorical_features else 'numerical'}
    for feat in top_features
]

# ------------------------------
# Routes
# ------------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict")
def predict_form():
    return render_template(
        "predict.html",
        features_info=features_info,
        categorical_values=categorical_values
    )

@app.route("/result", methods=["POST"])
def result():
    input_data = {}
    for feat in top_features:
        val = request.form.get(feat)
        if feat in form_categorical_features:
            input_data[feat] = val if val else ""
        else:
            input_data[feat] = float(val) if val else 0.0
    
    input_df = pd.DataFrame([input_data], columns=top_features)
    
    # Encode categorical features first
    for col in form_categorical_features:
        if col in input_df.columns and col in cat_to_encoder_idx:
            idx = cat_to_encoder_idx[col]
            input_df[col] = encoders[idx].transform(input_df[col].astype(str))
    
    # Reorder columns to match scaler's expected order
    scaler_feature_order = scaler.feature_names_in_.tolist()
    input_df_scaled = input_df[scaler_feature_order]
    
    # Scale all features
    input_df_scaled = pd.DataFrame(scaler.transform(input_df_scaled), columns=scaler_feature_order)
    
    # Reorder back to top_features order for model
    input_df = input_df_scaled[top_features]
    
    # Predict
    pred_prob = model.predict_proba(input_df)[:, 1][0]
    pred = model.predict(input_df)[0]

    probability = round(pred_prob, 4)
    will_transition = pred == 1
    
    return render_template(
        "result.html",
        will_transition=will_transition,
        probability=probability
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
