import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from dataset import ProcedureDataset
from model import LSTMAutoencoder
from config import *

dir = 'E:/OneDrive/Documentos/GitHub/iris/Iris_LSTM_procedimento/train_data/'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # Load the data
    df = pd.read_pickle(dir+ 'df.pkl')

    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y%m')

    # Sort by date
    df.sort_values('date', inplace=True)

    unique_procedures = df['codigo_procedimento'].unique()

    for procedure in unique_procedures:
        df_procedure = df[df['codigo_procedimento'] == procedure]

        # Create a Dataset
        dataset = ProcedureDataset(df_procedure[['x', 'y']].values, WINDOW_SIZE)

        # Create a DataLoader
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Create the model
        model = LSTMAutoencoder(2, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)

        # Loss function and optimizer
        criterion = nn.MSELoss()
# Continuação...

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training loop
        for epoch in range(NUM_EPOCHS):
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                output = model(x_batch)

                # Compute the loss
                loss = criterion(output, y_batch)

                # Backward pass
                loss.backward()

                # Update the weights
                optimizer.step()

            print(f'Epoch: {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item()}')

        # Save the model
        torch.save(model.state_dict(), f'model_{procedure}.pth')

        # Detect anomalies
        model.eval()
        predictions = []
        with torch.no_grad():
            for x_batch, _ in data_loader:
                x_batch = x_batch.to(DEVICE)
                output = model(x_batch)
                predictions.extend(output.cpu().numpy().tolist())

        # Calculate the error
        error = np.mean(np.power(df_procedure[['x', 'y']].values - predictions, 2), axis=1)

        # Get the quantile
        threshold = np.quantile(error, THRESHOLD)

        # Print the anomalies
        anomalies = df_procedure[error > threshold]
        print(f'Anomalies for procedure {procedure}:')
        print(anomalies)

if __name__ == '__main__':
    main()
