import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

print("Starting model training process...")

# Load dataset
file_path = "traffic_accident_dataset_20000-1.csv"
print(f"Loading dataset: {file_path}")
df = pd.read_csv(file_path)

print(f"Dataset loaded. Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")


print("Mapping Severity labels to numeric (1=Low, 2=Medium, 3=High, 4=Critical)...")
severity_mapper = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
df['Severity'] = df['Severity'].map(severity_mapper)
df['Severity'].fillna(1, inplace=True) # fallback just in case
print(f"Severity distribution:\n{df['Severity'].value_counts().sort_index()}")

# ─────────────────────────────────────────────────────────
# Features: new dataset columns
# ─────────────────────────────────────────────────────────
feature_cols = [
    'Road_Type', 'Road_Condition', 'Vehicle_Speed', 'Speed_Limit',
    'Vehicle_Type', 'Temperature', 'Humidity', 'Visibility',
    'Weather_Condition', 'Junction', 'Crossing', 'Railway_Crossing',
    'Stop_Sign', 'Speed_Breaker'
]

X = df[feature_cols].copy()
y = df['Severity']

# ─────────────────────────────────────────────────────────
# Encode categorical columns
# ─────────────────────────────────────────────────────────
encoders = {}
categorical_cols = ['Road_Type', 'Road_Condition', 'Vehicle_Type', 'Weather_Condition', 'Junction']

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le
    print(f"  Encoded '{col}': {list(le.classes_)}")

print(f"\nFinal feature set ({len(feature_cols)} features): {feature_cols}")
print(f"Dataset shape for training: {X.shape}")

# ─────────────────────────────────────────────────────────
# Train / Test split & model training
# ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Decision Tree Classifier...")
model = DecisionTreeClassifier(random_state=42, max_depth=12, min_samples_split=8)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"Total records used: {len(df)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("=" * 50)

# ─────────────────────────────────────────────────────────
# Save model artifacts
# ─────────────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.pkl')
joblib.dump(feature_cols, 'model/features.pkl')
joblib.dump(encoders, 'model/encoders.pkl')

print("\nModel, features, and encoders saved to 'model/' directory.")
print("Training complete!")
