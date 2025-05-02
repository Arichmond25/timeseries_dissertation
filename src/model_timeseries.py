
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load preprocessed time-series data
X_padded, y_padded, sequence_lengths = torch.load("timeseries_dataset.pt")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_padded = X_padded.to(device)
y_padded = y_padded.to(device)

# Define GRU-based sequence model
class GRUTimeSeriesPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.2):
        super(GRUTimeSeriesPredictor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    # Initialize model
    input_dim = X_padded.shape[2]
    hidden_dim = 64
    output_dim = y_padded.shape[2]
    model = GRUTimeSeriesPredictor(input_dim, hidden_dim, output_dim).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_padded)
        loss = criterion(outputs, y_padded)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), "timeseries_model.pth")
    print("Model saved as timeseries_model.pth")

    # Plot training loss
    plt.plot(range(1, num_epochs+1), train_losses)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
