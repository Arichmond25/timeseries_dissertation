import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

# Load data
df = pd.read_csv("parkinsons_updrs.data")

# Define usable features and targets
feature_columns = [
    'age', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
    'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA',
    'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
]
target_columns = ["motor_UPDRS", "total_UPDRS"]

# Age bins and sex map
age_bins = [(0, 50), (50, 65), (65, 150)]
age_labels = ["under_50", "50_65", "65_plus"]
sex_labels = {0: "female", 1: "male"}

# Normalize features and targets
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
df[feature_columns] = feature_scaler.fit_transform(df[feature_columns])
df[target_columns] = target_scaler.fit_transform(df[target_columns])

# Create synthetic interpolated sequences
os.makedirs("interpolated_data", exist_ok=True)
group_data = defaultdict(list)

for subject_id, subject_df in df.groupby("subject#"):
    subject_df = subject_df.sort_values("test_time")
    sex = subject_df["sex"].iloc[0]
    age = subject_df["age"].iloc[0]

    age_group = None
    for (low, high), label in zip(age_bins, age_labels):
        if low <= age < high:
            age_group = label
            break
    if age_group is None:
        continue

    group_key = f"{sex_labels[sex]}_{age_group}"
    feats = subject_df[feature_columns].values
    targets = subject_df[target_columns].values
    times = subject_df["test_time"].values

    if len(feats) < 2:
        continue  # can't interpolate

    # Create interpolated points between each pair
    for i in range(len(feats) - 1):
        mid_feat = (feats[i] + feats[i + 1]) / 2
        mid_targ = (targets[i] + targets[i + 1]) / 2
        delta_time = (times[i + 1] - times[i]) / 2

        interp_time = times[i] + delta_time
        group_data[group_key].append((interp_time, mid_feat, mid_targ))

# Save interpolated tensors
for group_key, data in group_data.items():
    if not data:
        continue
    data.sort(key=lambda x: x[0])
    times, feats, targs = zip(*data)
    feats_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
    targs_tensor = torch.tensor(targs, dtype=torch.float32).unsqueeze(0)
    lengths_tensor = torch.tensor([feats_tensor.shape[1]])
    save_path = os.path.join("interpolated_data", f"{group_key}_interpolated.pt")
    torch.save((feats_tensor, targs_tensor, lengths_tensor), save_path)
    print(f"Saved {save_path} with {feats_tensor.shape[1]} interpolated points")
