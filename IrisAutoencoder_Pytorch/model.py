import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score



class AutoencoderMLP(nn.Module):
    def __init__(self, input_shape, hidden_units):
        super(AutoencoderMLP, self).__init__()
        self.shared_encoder = nn.Linear(input_shape, hidden_units)
        self.shared_decoder = nn.Linear(hidden_units, input_shape)
        self.losses = []  # Para armazenar as perdas de cada época

    def forward(self, x):
        shared_encoded = self.shared_encoder(x)
        shared_decoded = self.shared_decoder(shared_encoded)
        return shared_decoded

    def train_model(self, x, device, batch_size, learning_rate, num_epochs, verbose=False):
        self.to(device)
        x = x.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        data_loader = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            epoch_losses = []  # Para armazenar as perdas de cada batch nesta época
            for data in data_loader:
                data = data.to(device)
                optimizer.zero_grad()
                outputs = self(data)
                loss = criterion(outputs, data)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            epoch_loss = sum(epoch_losses) / len(epoch_losses)  # Média das perdas nesta época
            self.losses.append(epoch_loss)  # Adicione a perda média desta época à lista

            if verbose:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.12f}")

    def get_loss(self, epoch):
        return self.losses[epoch]

    def evaluate(self, x, device):
        self.to(device)
        x = x.to(device)
        with torch.no_grad():
            reconstructed = self(x)
            loss = torch.mean((reconstructed - x) ** 2)
            print("Reconstruction Loss:", loss.item())

    def detect_anomalies(self, x, original_df, device, threshold):
        self.to(device)
        x = x.to(device)
        with torch.no_grad():
            reconstructed = self(x)
            losses = torch.mean((reconstructed - x) ** 2, dim=1)
            anomaly_indices = torch.where(losses > threshold)[0]
            anomaly_df = original_df.iloc[anomaly_indices.cpu().numpy()]
            anomaly_df['loss'] = losses[anomaly_indices].cpu().numpy()
            anomaly_df.to_csv('anomalies.csv', index=False)
            return anomaly_df
        
    def compute_and_save_metrics(self, x, device, metrics_df, epoch):
        self.to(device)
        x = x.to(device)
        with torch.no_grad():
            reconstructed = self(x)
            mse_loss = torch.mean((reconstructed - x) ** 2).item()
            mae_loss = mean_absolute_error(x.cpu().numpy(), reconstructed.cpu().numpy())
            r2 = r2_score(x.cpu().numpy(), reconstructed.cpu().numpy())
            explained_variance = explained_variance_score(x.cpu().numpy(), reconstructed.cpu().numpy())

        # Save the metrics to the DataFrame
        metrics_df.loc[epoch, "MSE"] = mse_loss
        metrics_df.loc[epoch, "MAE"] = mae_loss
        metrics_df.loc[epoch, "R2 Score"] = r2
        metrics_df.loc[epoch, "Explained Variance"] = explained_variance


