import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Load preprocessed time-series data
X_padded, y_padded, sequence_lengths = torch.load("timeseries_dataset.pt")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_padded = X_padded.to(device)
y_padded = y_padded.to(device)
sequence_lengths = sequence_lengths.to("cpu")  # needed for packing

# Improved GRU-based model
class GRUTimeSeriesPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(GRUTimeSeriesPredictor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed_input)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.layer_norm(out)
        return self.fc(out)

# Initialize model
input_dim = X_padded.shape[2]
hidden_dim = 64
output_dim = y_padded.shape[2]
model = GRUTimeSeriesPredictor(input_dim, hidden_dim, output_dim).to(device)

# Loss and optimizer
criterion = nn.MSELoss(reduction='none')  # we'll mask manually
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
train_losses = []
clip_value = 1.0

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_padded, sequence_lengths)
    
    # Mask padded values
    mask = torch.arange(outputs.shape[1])[None, :].to(device) < sequence_lengths[:, None].to(device)
    mask = mask.unsqueeze(2).expand_as(outputs).float()
    
    loss_matrix = criterion(outputs, y_padded)
    masked_loss = (loss_matrix * mask).sum() / mask.sum()
    
    masked_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    optimizer.step()

    train_losses.append(masked_loss.item())

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Masked Loss: {masked_loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "timeseries_model_improved.pth")
print("Improved model saved as timeseries_model_improved.pth")

# Plot training loss
plt.plot(range(1, num_epochs+1), train_losses)
plt.title("Training Loss Over Epochs (Improved GRU)")
plt.xlabel("Epoch")
plt.ylabel("Masked MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
