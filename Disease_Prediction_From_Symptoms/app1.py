# Page config MUST be set before anything else from Streamlit
import streamlit as st
st.set_page_config(page_title="Disease Predictor Chatbot", page_icon="🩺", layout="centered")

# Other imports
import joblib
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import sys


# Load Model and Tools
@st.cache_resource
def load_model_and_tools():
    try:
        clf = joblib.load("logistic_regression_grid_model.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        suggestions_df = pd.read_csv("disease_symptoms_tips.csv")
        suggestion_map = dict(zip(suggestions_df['Disease'], suggestions_df['Tips']))
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased")
        bert_model.eval()
        return clf, label_encoder, suggestion_map, tokenizer, bert_model
    except Exception as e:
        st.error(f"Error loading model/tools: {e}")
        raise

clf, label_encoder, suggestion_map, tokenizer, bert_model = load_model_and_tools()

# BERT Embedding

def get_bert_embedding(text):
    try:
        if not text.strip():
            raise ValueError("Input text is empty.")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return embedding
    except Exception as e:
        st.error(f"Error generating BERT embedding: {e}")
        raise


# Predict Top Disease (only one)

def predict_top_diseases(symptom_text, top_n=1):
    try:
        features = get_bert_embedding(symptom_text.lower()).reshape(1, -1)
        probs = clf.predict_proba(features).flatten()
        top_indices = np.argsort(probs)[::-1][:top_n]
        top_diseases = label_encoder.inverse_transform(top_indices)
        top_probs = probs[top_indices]
        return list(zip(top_diseases, top_probs))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return []

# Session State Setup

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Disease Predictor Bot. Tell me your symptoms and I'll try to guess the most likely disease and give you health tips."}
    ]


# Display Chat Messages

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input and Response

user_input = st.chat_input("Type your symptoms here...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing symptoms..."):
            top_results = predict_top_diseases(user_input, top_n=1)

        if top_results:
            disease, prob = top_results[0]
            advice = suggestion_map.get(disease, "No suggestions available.")
            response = (
                f"**I believe you might have:** {disease}\n"
                f"- **Confidence:** {prob:.2%}\n"
                f"- **Advice:** {advice}"
            )
        else:
            response = "Sorry, I couldn't make a prediction. Please try describing your symptoms more clearly."

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
