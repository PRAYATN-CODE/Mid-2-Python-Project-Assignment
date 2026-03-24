import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("Loading card_transdata.csv...")
# Load the dataset
df_full = pd.read_csv('card_transdata.csv')

# Take a 10% random sample to ensure the script runs quickly on a standard laptop
# (1 million rows + SMOTE + Random Forest = extremely slow otherwise)
df = df_full.sample(frac=0.1, random_state=42).reset_index(drop=True)

print(f"Processing {len(df)} transactions...")

# Separate features and target
X = df.drop('fraud', axis=1)
y = df['fraud']

# Feature Selection (Evaluating Importance)
print("Evaluating Feature Importance...")
rf_selector = RandomForestClassifier(n_estimators=20, random_state=42)
rf_selector.fit(X, y)

# Print feature importances
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_selector.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importances:\n", importances)

# Since there are only 7 features and all are interpretable, we will keep them all for the UI
selected_features = X.columns.tolist()

# Train-Test Split (Crucial: Split BEFORE SMOTE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nScaling Data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handling Imbalanced Data (SMOTE)
print(f"\nOriginal Training Target Distribution:\n{y_train.value_counts()}")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f"SMOTE Training Target Distribution:\n{y_train_smote.value_counts()}")

# Model Training & Comparison
print("\nTraining multiple models for comparison...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
}

best_model = None
best_auc = 0

for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n--- {name} ---")
    print(f"ROC-AUC Score: {auc:.4f}")
    
    if auc > best_auc:
        best_auc = auc
        best_model = model

print(f"\nBest Model Selected: {type(best_model).__name__} with AUC: {best_auc:.4f}")

# Save Artifacts for Deployment
print("\nSaving model, scaler, and feature names...")
artifacts = {
    'model': best_model,
    'scaler': scaler,
    'features': selected_features
}

with open('fraud_detection_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("Success! Run the Streamlit app to launch the simulation.")