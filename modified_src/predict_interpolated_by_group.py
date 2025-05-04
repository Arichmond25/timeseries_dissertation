import torch
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from gru_model import GRUTimeSeriesPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load scalers
target_scaler = joblib.load("target_scaler.pkl")

# Directory with interpolated .pt files
data_dir = "interpolated_data"
model_dir = "trained_models"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluate each file
for file in os.listdir(data_dir):
    if not file.endswith(".pt"):
        continue

    group_key = file.replace("_interpolated.pt", "")
    model_path = os.path.join(model_dir, f"{group_key}_gru.pth")
    if not os.path.exists(model_path):
        print(f"Model for {group_key} not found. Skipping.")
        continue

    # Load data
    X, y, lengths = torch.load(os.path.join(data_dir, file))
    X, y = X.to(device), y.to(device)

    # Load model
    input_dim = X.shape[2]
    model = GRUTimeSeriesPredictor(input_dim=input_dim, hidden_dim=64, output_dim=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Predict
    with torch.no_grad():
        preds = model(X, lengths).squeeze(0).cpu().numpy()
        targets = y.squeeze(0).cpu().numpy()

    # Inverse scale
    preds_inv = target_scaler.inverse_transform(preds)
    targets_inv = target_scaler.inverse_transform(targets)

    # Metrics
    mse = mean_squared_error(targets_inv, preds_inv)
    mae = mean_absolute_error(targets_inv, preds_inv)
    r2 = r2_score(targets_inv, preds_inv)

    print(f"Group: {group_key}")
    print(f"  MSE: {mse:.2f} | MAE: {mae:.2f} | R²: {r2:.2f}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(preds_inv[:, 0], label="Predicted Motor UPDRS", marker='x')
    plt.plot(targets_inv[:, 0], label="True Motor UPDRS", marker='o')
    plt.plot(preds_inv[:, 1], label="Predicted Total UPDRS", linestyle='--', marker='x')
    plt.plot(targets_inv[:, 1], label="True Total UPDRS", linestyle='--', marker='o')
    plt.title(f"Interpolated Prediction: {group_key}")
    plt.xlabel("Time Step")
    plt.ylabel("UPDRS Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
