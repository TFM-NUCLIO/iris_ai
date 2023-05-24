import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model import AutoencoderMLP
import config

# Carregar dados
df_X1 = pd.read_pickle(config.df_X1_path)
df_X2 = pd.read_pickle(config.df_X2_path)
df_X3 = pd.read_pickle(config.df_X3_path)

# Converter para tensor do PyTorch
X1_tensor = torch.tensor(df_X1.values, dtype=torch.float)
X2_tensor = torch.tensor(df_X2.values, dtype=torch.float)
X3_tensor = torch.tensor(df_X3.values, dtype=torch.float)

# Definir configurações
input_shape = X1_tensor.shape[1]  # Ou X2_tensor.shape[1] ou X3_tensor.shape[1], dependendo do tamanho
hidden_units = config.hidden_units
learning_rate = config.learning_rate
batch_size = config.batch_size
num_epochs = config.num_epochs

# Criar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoencoderMLP(input_shape, hidden_units).to(device)

# Treinar modelo
model.train_model(X1_tensor, device=device, batch_size=batch_size, learning_rate=learning_rate, num_epochs=num_epochs, verbose=True)
model.train_model(X2_tensor, device=device, batch_size=batch_size, learning_rate=learning_rate, num_epochs=num_epochs, verbose=True)
model.train_model(X3_tensor, device=device, batch_size=batch_size, learning_rate=learning_rate, num_epochs=num_epochs, verbose=True)

# Avaliar modelo
model.evaluate(X1_tensor, device)
model.evaluate(X2_tensor, device)
model.evaluate(X3_tensor, device)

# Detectar anomalias
anomaly_indices_X1 = model.detect_anomalies(X1_tensor.to(device), df_X1, device, threshold=config.threshold)
anomaly_indices_X2 = model.detect_anomalies(X2_tensor.to(device), df_X2, device, threshold=config.threshold)
anomaly_indices_X3 = model.detect_anomalies(X3_tensor.to(device), df_X3, device, threshold=config.threshold)

print("Anomaly Indices (X1):", anomaly_indices_X1)
print("Anomaly Indices (X2):", anomaly_indices_X2)
print("Anomaly Indices (X3):", anomaly_indices_X3)

# Criar um DataFrame vazio para armazenar as métricas
metrics_df = pd.DataFrame(columns=["Dataset", "Epoch", "Loss"])

# Loop de treinamento e registro das métricas
for dataset_name, dataset_tensor in zip(["X1", "X2", "X3"], [X1_tensor, X2_tensor, X3_tensor]):
    # Criar modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoencoderMLP(input_shape, hidden_units).to(device)

    # Treinar modelo
    model.train_model(dataset_tensor, device=device, batch_size=batch_size, learning_rate=learning_rate, num_epochs=num_epochs, verbose=True)

    # Avaliar modelo
    model.evaluate(dataset_tensor, device)

    # Detectar anomalias
    anomaly_indices = model.detect_anomalies(dataset_tensor.to(device), dataset_name, device, threshold=config.threshold)
    
    # Registrar métricas no DataFrame
    for epoch in range(num_epochs):
        loss = model.get_loss(epoch)
        metrics_df = metrics_df.append({"Dataset": dataset_name, "Epoch": epoch + 1, "Loss": loss.item()}, ignore_index=True)

# Salvar DataFrame em um arquivo CSV
metrics_df.to_csv("metrics.csv", index=False)