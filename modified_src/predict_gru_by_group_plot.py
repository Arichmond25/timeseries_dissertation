import torch
import joblib
import pandas as pd
import numpy as np
from gru_model import GRUTimeSeriesPredictor
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
import os
import matplotlib.pyplot as plt

# Load scalers and config
feature_scaler = joblib.load("feature_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Define stratification bins
def get_age_group(age):
    if age < 50:
        return "under_50"
    elif age < 65:
        return "50_65"
    else:
        return "65_plus"

sex_map = {0: "male", 1: "female"}

# Load new patient CSV (must include age, sex, test_time, and features)
new_df = pd.read_csv("new_patient_data.csv")
new_df = new_df.sort_values(by="test_time")

# Extract stratification
sex = sex_map[int(new_df["sex"].iloc[0])]
age = new_df["age"].iloc[0]
age_group = get_age_group(age)
group_key = f"{sex}_{age_group}"

# Prepare input features
features = new_df[feature_columns].copy()
features = feature_scaler.transform(features)

# Add delta time
delta_time = new_df["test_time"].diff().fillna(0).values.reshape(-1, 1)
delta_time = MinMaxScaler().fit_transform(delta_time)
features = np.concatenate([features, delta_time], axis=1)

# Convert to tensor
input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
length_tensor = torch.tensor([input_tensor.shape[1]])

# Load model
model_path = os.path.join("trained_models", f"{group_key}_gru.pth")
input_dim = input_tensor.shape[2]
model = GRUTimeSeriesPredictor(input_dim=input_dim, hidden_dim=64, output_dim=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Predict
with torch.no_grad():
    output = model(input_tensor, length_tensor)
    output_np = output.squeeze(0).numpy()
    output_inv = target_scaler.inverse_transform(output_np)

# Print predicted UPDRS
print(f"Predicted UPDRS for {group_key}:")
for t, (motor, total) in enumerate(output_inv):
    print(f"  Step {t+1}: Motor={motor:.2f}, Total={total:.2f}")

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(output_inv[:, 0], label="Predicted Motor UPDRS", marker='o')
plt.plot(output_inv[:, 1], label="Predicted Total UPDRS", marker='x')
plt.title(f"Predicted UPDRS Scores for {group_key}")
plt.xlabel("Time Step")
plt.ylabel("UPDRS Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
