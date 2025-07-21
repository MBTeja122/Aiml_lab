from flask import Flask, render_template, request, jsonify
import joblib, pandas as pd, numpy as np, json, random, torch
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load Models and Data

try:
    clf = joblib.load("logistic_model_sbert1.pkl")
    label_encoder = joblib.load("label_encoder_sbert1.pkl")
    disease_info_df = pd.read_csv("disease_info.csv")
    description_map = dict(zip(disease_info_df["Disease"], disease_info_df["Description"]))
    suggestion_map = dict(zip(disease_info_df["Disease"], disease_info_df["Tips"]))
except Exception as e:
    print(f"❗ Error loading disease model or files: {e}")
    clf, label_encoder, description_map, suggestion_map = None, None, {}, {}

try:
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"❗ Error loading SBERT: {e}")
    sbert_model = None

def get_sbert_embedding(text):
    if sbert_model is None:
        return np.zeros(384)
    return sbert_model.encode([text], convert_to_numpy=True)[0]

def predict_top_diseases(symptom_text, top_n=1):
    if clf is None or label_encoder is None:
        return []
    features = get_sbert_embedding(symptom_text.lower()).reshape(1, -1)
    probs = clf.predict_proba(features).flatten()
    top_indices = np.argsort(probs)[::-1][:top_n]
    top_diseases = label_encoder.inverse_transform(top_indices)
    top_probs = probs[top_indices]
    return list(zip(top_diseases, top_probs))

#routers
@app.route("/")
def index():
    return render_template("index.html") # into html

@app.route("/chat")
def chat():
    return render_template("chat.html")  # Chatbot html

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"type": "error", "response": " Please enter your symptoms."})

    lower_msg = user_message.lower()

    # Special case: if user says only "hi"
    if lower_msg == "hi":
        return jsonify({
            "type": "response",
            "response": "Hi, I can assist you."
        })

    # Disease prediction
    result = predict_top_diseases(user_message, top_n=1)
    if result:
        disease, prob = result[0]
        desc = description_map.get(disease, "No description available.")
        tips = suggestion_map.get(disease, "No tips available.").split(";")

        return jsonify({
            "type": "disease",
            "disease": disease,
            "confidence": float(f"{prob:.4f}"),
            "description": desc,
            "tips": [tip.strip() for tip in tips]
        })

    # Final fallback
    return jsonify({
        "type": "fallback",
        "response": "I'm sorry, I didn't quite understand that. Please enter your symptoms clearly so I can help."
    })


@app.route("/autocomplete")
def autocomplete():
    try:
        with open("symptom_autocomplete.json", "r") as f:
            return jsonify(json.load(f))
    except:
        return jsonify([])


# Run App
if __name__ == "__main__":
    app.run(debug=True)
