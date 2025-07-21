import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

#Load and preprocess data
df = pd.read_csv("Training.csv")
df['Symptoms'] = df['Symptoms'].str.replace(r';', ' ', regex=True).str.lower()

#Encode labels
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Disease'])

#Load Sentence-BERT model (384-d embedding)
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

#Get SBERT embeddings
print("Generating Sentence-BERT embeddings...")
X = sbert_model.encode(df['Symptoms'].tolist(), show_progress_bar=True)
y = df['Label'].values

#Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Grid search for logistic regression
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
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
all_labels = list(range(len(label_encoder.classes_)))
#Evaluation
y_pred = best_model.predict(X_test)
print("\n✅ Best Parameters Found:")
print(grid_search.best_params_)
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=all_labels,
    target_names=label_encoder.classes_,
    zero_division=0  
))
print(" Accuracy:", accuracy_score(y_test, y_pred))

# Saving model and label encoder
joblib.dump(best_model, 'logistic_model_sbert1.pkl')
joblib.dump(label_encoder, 'label_encoder_sbert1.pkl')
print("\nSBERT-based model and label encoder saved.")
