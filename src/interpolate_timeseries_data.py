import torch
import numpy as np

# Load the original dataset
X_padded, y_padded, sequence_lengths = torch.load("timeseries_dataset.pt")

# Function to interpolate between two tensors
def interpolate_sequences(seq1, seq2, alpha=0.5):
    return alpha * seq1 + (1 - alpha) * seq2

# Generate interpolated samples (pairwise)
interpolated_X = []
interpolated_y = []
interpolated_lengths = []

for i in range(0, len(X_padded) - 1, 2):
    x_interp = interpolate_sequences(X_padded[i], X_padded[i + 1], alpha=0.5)
    y_interp = interpolate_sequences(y_padded[i], y_padded[i + 1], alpha=0.5)

    interpolated_X.append(x_interp)
    interpolated_y.append(y_interp)
    interpolated_lengths.append(max(sequence_lengths[i], sequence_lengths[i + 1]))

# Combine with original for testing if needed
X_aug = torch.stack(interpolated_X)
y_aug = torch.stack(interpolated_y)
sequence_lengths_aug = torch.tensor(interpolated_lengths)

# Save the interpolated dataset
torch.save((X_aug, y_aug, sequence_lengths_aug), "interpolated_timeseries_dataset.pt")
print(f"Interpolated {len(X_aug)} sequences saved to interpolated_timeseries_dataset.pt")
