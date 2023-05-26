import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMAutoencoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Encode
        _, (hidden, _) = self.lstm(x)

        # Repeat the hidden state for as many timesteps as we have
        hidden_repeated = hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2)

        # Decode
        out = self.linear(hidden_repeated)

        return out

    def train_model(self, dataset, device, batch_size, learning_rate, num_epochs):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            for data in data_loader:
                data = data.to(device)
                optimizer.zero_grad()
                outputs = self(data)
                loss = criterion(outputs, data)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    
    def evaluate(self, dataset, device):
        # Similar to train_model, but without the optimization step.
        # You might want to calculate and print the loss here too.
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        self.eval()  # Set the model to evaluation mode
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
                        # Get the relevant anomaly information
                        cnes = dataset[i, 0, 0]  # Example: cnes at position [0, 0] of the input tensor
                        codigo_procedimento = dataset[i, 0, 1]  # Example: procedure code at position [0, 1] of the input tensor
                        anomalia = {
                            'cnes': cnes.item(),
                            'codigo_procedimento': codigo_procedimento.item(),
                            'anomalia_data': []  # To store the dates of the anomalies
                        }

                        # Add the dates of the anomalies
                        for j, value in enumerate(data[i, :, 2:].flatten(), start=201001):
                            if outputs[i, :, j - 201001] > threshold:
                                anomalia['anomalia_data'].append(j)

                        anomalies.append(anomalia)

        return anomalies
