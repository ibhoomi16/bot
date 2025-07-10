import os
import pickle
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from client import load_partition_data
from xgboost import XGBClassifier

# --- Step 1: Load global feature names from FL round 20 ---
with open('global_models/global_model_params_round_20.pkl', 'rb') as f:
    global_params = pickle.load(f)

feature_names = global_params['feature_names']
print(f"✅ Loaded {len(feature_names)} global features from round 20.")

# --- Step 2: Load training data from a client (e.g., client_1) ---
clients = ['client_1', 'client_2', 'client_3']
X_list, y_list = [], []
for client in clients:
    Xc, yc, _, _ = load_partition_data(client, 'train')
    X_list.append(Xc)
    y_list.append(yc)
X = pd.concat(X_list, ignore_index=True)
y = pd.concat(y_list, ignore_index=True)

print("Train shape:", X.shape, y.shape)
print("First 10 train session IDs:", X.index[:10])

# --- Step 3: Align features ---
X = X[feature_names]  # Keep only global FL features

# --- Step 4: Train classifier ---
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X, y)
print(f"✅ Trained XGBClassifier on {X.shape[0]} samples.")

# --- Step 5: Save model to disk ---
os.makedirs('model', exist_ok=True)  # Ensure model/ directory exists
joblib.dump(model, 'model/global_model.joblib')
print("✅ Final model saved to 'model/global_model.joblib' for use in Flask API.")
