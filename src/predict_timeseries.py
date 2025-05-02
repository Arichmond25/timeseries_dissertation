
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from model_timeseries import GRUTimeSeriesPredictor

# Load scalers and column order
feature_scaler = joblib.load("feature_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")
column_order = joblib.load("feature_columns.pkl")

# Load the full raw dataset (CSV must be in the same directory)
raw_data = pd.read_csv("parkinsons_updrs.data")

# Select one subject
subject_id = 1
subject_data = raw_data[raw_data['subject#'] == subject_id].sort_values(by='test_time')

# Extract features and targets
features = subject_data[column_order]
true_updrs = subject_data[['motor_UPDRS', 'total_UPDRS']].values
time_points = subject_data['test_time'].values

# Scale features
scaled_features = feature_scaler.transform(features)
X_subject = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0)  # shape: (1, time, features)

# Load model
input_dim = X_subject.shape[2]
hidden_dim = 64
output_dim = 2
model = GRUTimeSeriesPredictor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
model.load_state_dict(torch.load("timeseries_model.pth"))
model.eval()

# Predict
with torch.no_grad():
    y_pred = model(X_subject).squeeze(0).numpy()
    y_pred_inverse = target_scaler.inverse_transform(y_pred)

# Plot true vs predicted UPDRS over time
plt.figure(figsize=(12, 6))
plt.plot(time_points, true_updrs[:, 0], label='True Motor UPDRS', marker='o')
plt.plot(time_points, y_pred_inverse[:, 0], label='Predicted Motor UPDRS', marker='x')
plt.plot(time_points, true_updrs[:, 1], label='True Total UPDRS', marker='o', linestyle='--')
plt.plot(time_points, y_pred_inverse[:, 1], label='Predicted Total UPDRS', marker='x', linestyle='--')

plt.title(f"UPDRS Predictions Over Time (Subject #{subject_id})")
plt.xlabel("Test Time")
plt.ylabel("UPDRS Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
