import joblib

# Load the model
model = joblib.load("model/global_model.joblib")

# Print feature names
try:
    print("✅ Features used during training:")
    print(list(model.feature_names_in_))
except AttributeError:
    print("❌ This model does not have feature_names_in_. It may not be a sklearn-compatible model.")
