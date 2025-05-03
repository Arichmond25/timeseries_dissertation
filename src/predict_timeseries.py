
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.nn.utils.rnn import pad_sequence
from model_timeseries import GRUTimeSeriesPredictor

# Load scalers and config
target_scaler = joblib.load("target_scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Load dataset
X_padded, y_padded, sequence_lengths = torch.load("timeseries_dataset.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_padded = X_padded.to(device)
y_padded = y_padded.to(device)

# Load model
input_dim = X_padded.shape[2]
hidden_dim = 64
output_dim = 2
model = GRUTimeSeriesPredictor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
model.load_state_dict(torch.load("timeseries_model_improved.pth"))
model.eval()

# Predict
with torch.no_grad():
    predictions = model(X_padded, sequence_lengths)
    y_pred = predictions.cpu().numpy()
    y_true = y_padded.cpu().numpy()
    sequence_lengths = sequence_lengths.cpu().numpy()

    # Inverse scale
    y_pred_inv = target_scaler.inverse_transform(y_pred.reshape(-1, 2)).reshape(y_pred.shape)
    y_true_inv = target_scaler.inverse_transform(y_true.reshape(-1, 2)).reshape(y_true.shape)

# Compute metrics over valid (unpadded) values
true_all = []
pred_all = []

for i in range(len(sequence_lengths)):
    valid_len = sequence_lengths[i]
    true_all.append(y_true_inv[i][:valid_len])
    pred_all.append(y_pred_inv[i][:valid_len])

true_all = np.concatenate(true_all)
pred_all = np.concatenate(pred_all)

mae = mean_absolute_error(true_all, pred_all)
mse = mean_squared_error(true_all, pred_all)
r2 = r2_score(true_all, pred_all)

print(f"Evaluation on Full Dataset:")
print(f"  MAE: {mae:.2f}")
print(f"  MSE: {mse:.2f}")
print(f"  R²:  {r2:.2f}")

# Plot random 5 sequences
num_to_plot = min(5, len(sequence_lengths))
indices = np.random.choice(len(sequence_lengths), size=num_to_plot, replace=False)

for i in indices:
    time_steps = list(range(sequence_lengths[i]))
    true_seq = y_true_inv[i][:sequence_lengths[i]]
    pred_seq = y_pred_inv[i][:sequence_lengths[i]]

    plt.figure(figsize=(12, 5))
    plt.plot(time_steps, true_seq[:, 0], label="True Motor UPDRS", marker="o")
    plt.plot(time_steps, pred_seq[:, 0], label="Pred Motor UPDRS", marker="x")
    plt.plot(time_steps, true_seq[:, 1], label="True Total UPDRS", marker="o", linestyle="--")
    plt.plot(time_steps, pred_seq[:, 1], label="Pred Total UPDRS", marker="x", linestyle="--")
    plt.title(f"Prediction vs Ground Truth (Sequence {i})")
    plt.xlabel("Time Step")
    plt.ylabel("UPDRS Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
