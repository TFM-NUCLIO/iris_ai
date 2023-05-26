import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMAutoencoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden_repeated = hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        out = self.linear(hidden_repeated)
        return out

    def train_model(self, dataset, device, batch_size, learning_rate, num_epochs, num_months):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []

        for epoch in range(num_epochs):
            for i, data in enumerate(data_loader):
                if i * batch_size >= num_months:
                    break
                data = data.to(device)
                optimizer.zero_grad()
                outputs = self(data)
                loss = criterion(outputs, data)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title("Training Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def evaluate(self, dataset, device):
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        self.eval()
        with torch.no_grad():
            for data in data_loader:
                data = data.to(device)
                outputs = self(data)

    def detect_anomalies(self, dataset, device, threshold):
        self.to(device)
        dataset = dataset.to(device)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        anomalies = []

        with torch.no_grad():
            self.eval()

            for data in data_loader:
                outputs = self(data)
                recon_loss = torch.mean((outputs - data) ** 2, dim=(1, 2))

                for i, loss in enumerate(recon_loss):
                    if loss > threshold:
                        anomalia = {
                            'anomalia_data': []
                        }

                        for j, value in enumerate(data[i, :, :].flatten(), start=201001):
                            if outputs[i, :, j - 201001] > threshold:
                                anomalia['anomalia_data'].append(j)

                        anomalies.append(anomalia)

        return anomalies
