import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Model definition
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
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        unpacked, _ = pad_packed_sequence(out, batch_first=True)
        normed = self.layer_norm(unpacked)
        return self.fc(normed)

if __name__ == "__main__":
    # Training parameters
    hidden_dim = 64
    output_dim = 2
    num_epochs = 100
    lr = 0.001
    clip_value = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training loop per file
    os.makedirs("trained_models", exist_ok=True)
    data_dir = "stratified_data"
    for file in os.listdir(data_dir):
        if not file.endswith(".pt"):
            continue

        group_name = file.replace(".pt", "")
        print(f"Training model for group: {group_name}")

        X, y, lengths = torch.load(os.path.join(data_dir, file))
        X, y = X.to(device), y.to(device)

        model = GRUTimeSeriesPredictor(input_dim=X.shape[2], hidden_dim=hidden_dim, output_dim=output_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss(reduction="none")

        train_losses = []

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X, lengths)

            # Mask padded timesteps
            mask = torch.arange(X.shape[1])[None, :].to(device) < lengths[:, None].to(device)
            mask = mask.unsqueeze(2).expand_as(output).float()
            loss_matrix = criterion(output, y)
            masked_loss = (loss_matrix * mask).sum() / mask.sum()

            masked_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            train_losses.append(masked_loss.item())

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {masked_loss.item():.4f}")

        # Save model
        model_path = os.path.join("trained_models", f"{group_name}_gru.pth")
        torch.save(model.state_dict(), model_path)
        print(f"  Saved model to {model_path}\n")

        # Plot loss
        plt.figure()
        plt.plot(range(1, num_epochs+1), train_losses)
        plt.title(f"Training Loss: {group_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Masked MSE Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join("trained_models", f"{group_name}_loss.png"))
        plt.close()
