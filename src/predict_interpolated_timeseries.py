
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
from model_timeseries import GRUTimeSeriesPredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load interpolated data
X_interp, y_interp, sequence_lengths = torch.load("interpolated_timeseries_dataset.pt")

# Load scalers and feature config
target_scaler = joblib.load("target_scaler.pkl")

# Load model
input_dim = X_interp.shape[2]
hidden_dim = 64
output_dim = 2
model = GRUTimeSeriesPredictor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
model.load_state_dict(torch.load("timeseries_model.pth"))
model.eval()

# Predict
with torch.no_grad():
    y_pred = model(X_interp).numpy()
    y_pred_inverse = target_scaler.inverse_transform(y_pred.reshape(-1, 2)).reshape(y_pred.shape)
    y_true_inverse = target_scaler.inverse_transform(y_interp.reshape(-1, 2)).reshape(y_interp.shape)

# Evaluate on full sequences
mae = np.mean(np.abs(y_pred_inverse - y_true_inverse))
mse = np.mean((y_pred_inverse - y_true_inverse) ** 2)
print(f"Interpolated Data Evaluation:")
print(f"  MAE: {mae:.2f}")
print(f"  MSE: {mse:.2f}")

# Plot one interpolated sequence
seq_index = 0
time_steps = list(range(sequence_lengths[seq_index].item()))
true_seq = y_true_inverse[seq_index][:len(time_steps)]
pred_seq = y_pred_inverse[seq_index][:len(time_steps)]

plt.figure(figsize=(12, 6))
plt.plot(time_steps, true_seq[:, 0], label="True Motor UPDRS", marker="o")
plt.plot(time_steps, pred_seq[:, 0], label="Pred Motor UPDRS", marker="x")
plt.plot(time_steps, true_seq[:, 1], label="True Total UPDRS", marker="o", linestyle="--")
plt.plot(time_steps, pred_seq[:, 1], label="Pred Total UPDRS", marker="x", linestyle="--")
plt.title("UPDRS Predictions on Interpolated Sequence")
plt.xlabel("Time Step")
plt.ylabel("UPDRS Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
