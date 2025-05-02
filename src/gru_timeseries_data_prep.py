
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib
from torch.nn.utils.rnn import pad_sequence

# Load the raw dataset from the manually downloaded CSV
raw_data = pd.read_csv("parkinsons_updrs.data")

# Updated feature columns to match actual dataset
feature_columns = [
    'age', 'sex', 'test_time', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
    'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA',
    'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
]

target_columns = ['motor_UPDRS', 'total_UPDRS']

# Extract features, targets, and subject identifiers
features = raw_data[feature_columns]
targets = raw_data[target_columns]
subjects = raw_data['subject#']

# Save feature column order
joblib.dump(feature_columns, "feature_columns.pkl")

# Fit scalers and save
feature_scaler = MinMaxScaler().fit(features)
target_scaler = MinMaxScaler().fit(targets)
joblib.dump(feature_scaler, "feature_scaler.pkl")
joblib.dump(target_scaler, "target_scaler.pkl")

# Apply scaling
scaled_features = feature_scaler.transform(features)
scaled_targets = target_scaler.transform(targets)

# Prepare full DataFrame with metadata
scaled_df = pd.DataFrame(scaled_features, columns=feature_columns)
scaled_df['subject#'] = subjects
scaled_df['motor_UPDRS'] = scaled_targets[:, 0]
scaled_df['total_UPDRS'] = scaled_targets[:, 1]
scaled_df['test_time'] = raw_data['test_time']

# Group by subject and sort by test_time
sequence_data = []
sequence_targets = []

for subject_id, group in scaled_df.groupby('subject#'):
    group = group.sort_values(by='test_time')
    feature_seq = group[feature_columns].values
    target_seq = group[['motor_UPDRS', 'total_UPDRS']].values

    if len(feature_seq) >= 2:
        sequence_data.append(torch.tensor(feature_seq, dtype=torch.float32))
        sequence_targets.append(torch.tensor(target_seq, dtype=torch.float32))

# Pad sequences
X_padded = pad_sequence(sequence_data, batch_first=True)
y_padded = pad_sequence(sequence_targets, batch_first=True)
sequence_lengths = torch.tensor([len(seq) for seq in sequence_data])

# Save time-series dataset
torch.save((X_padded, y_padded, sequence_lengths), "timeseries_dataset.pt")
print(f"Saved timeseries_dataset.pt with shape {X_padded.shape}")
