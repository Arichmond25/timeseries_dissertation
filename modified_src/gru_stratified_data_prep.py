import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

# Load data
df = pd.read_csv("parkinsons_updrs.data")

# Define feature and target columns
feature_columns = [
    'age', 'sex', 'test_time', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
    'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA',
    'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
]
target_columns = ["motor_UPDRS", "total_UPDRS"]

# Define age groups and sex categories
age_bins = [(0, 50), (50, 65), (65, 150)]
age_labels = ["under_50", "50_65", "65_plus"]
sex_labels = {0: "male", 1: "female"}

# Create output directory
os.makedirs("stratified_data", exist_ok=True)

# Initialize scalers
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Normalize features and targets globally
df[feature_columns] = feature_scaler.fit_transform(df[feature_columns])
df[target_columns] = target_scaler.fit_transform(df[target_columns])

# Save scalers
import joblib
joblib.dump(feature_scaler, "feature_scaler.pkl")
joblib.dump(target_scaler, "target_scaler.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

# Create stratified datasets
group_data = defaultdict(list)

# Group by subject
for subject_id, subject_df in df.groupby("subject#"):
    subject_df = subject_df.sort_values(by="test_time")
    sex = subject_df["sex"].iloc[0]
    age = subject_df["age"].iloc[0]
    age_group = None
    for (lower, upper), label in zip(age_bins, age_labels):
        if lower <= age < upper:
            age_group = label
            break
    if age_group is None:
        continue

    # Compute delta time (normalized)
    delta_time = subject_df["test_time"].diff().fillna(0).values.reshape(-1, 1)
    delta_time = MinMaxScaler().fit_transform(delta_time)

    features = subject_df[feature_columns].values
    features = np.concatenate([features, delta_time], axis=1)  # Append delta_time
    targets = subject_df[target_columns].values

    group_key = f"{sex_labels[sex]}_{age_group}"
    group_data[group_key].append((torch.tensor(features, dtype=torch.float32),
                                  torch.tensor(targets, dtype=torch.float32)))

# Pad and save sequences per group
for group_key, sequences in group_data.items():
    X_seq, y_seq = zip(*sequences)
    X_padded = torch.nn.utils.rnn.pad_sequence(X_seq, batch_first=True)
    y_padded = torch.nn.utils.rnn.pad_sequence(y_seq, batch_first=True)
    lengths = torch.tensor([len(seq) for seq in X_seq])
    torch.save((X_padded, y_padded, lengths), f"stratified_data/{group_key}.pt")
    print(f"Saved {group_key}.pt with {len(X_seq)} sequences")
