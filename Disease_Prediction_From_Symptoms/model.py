import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load and preprocess data
df = pd.read_csv("Data.csv")
df['Symptoms'] = df['Symptoms'].str.replace(r';', ' ', regex=True).str.lower()

label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Disease'])

# 2. Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS token

# 3. Generate BERT embeddings
print("Generating BERT embeddings...")
X = np.vstack([get_bert_embedding(text) for text in df['Symptoms']])
y = df['Label'].values

# 4. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Grid Search for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],  # 'l1' if using 'liblinear' or 'saga'
    'solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("\nBest Parameters Found:")
print(grid_search.best_params_)

# 6. Evaluate model
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 7. Load treatment suggestions
suggestions_df = pd.read_csv("disease_symptoms_tips.csv")
suggestion_map = dict(zip(suggestions_df['Disease'], suggestions_df['Tips']))

# 8. Prediction function
def predict_and_suggest(symptom_text):
    symptom_text = symptom_text.lower()
    features = get_bert_embedding(symptom_text).reshape(1, -1)
    pred = best_model.predict(features)[0]
    disease = label_encoder.inverse_transform([pred])[0]
    suggestions = suggestion_map.get(disease, "No suggestions available.")
    return disease, suggestions

# 9. Test Example
user_input = "frequent urination, blurry vision, dry mouth"
disease, advice = predict_and_suggest(user_input)
print("\nUser Symptoms:", user_input)
print("Predicted Disease:", disease)
print("Suggestions:\n", advice)

# 10. Save model and encoder
joblib.dump(best_model, 'logistic_regression_grid_model3.pkl')
joblib.dump(label_encoder, 'label_encoder3.pkl')
print("\nTuned Logistic Regression model and label encoder saved.")
