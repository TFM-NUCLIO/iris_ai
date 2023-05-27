import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class ProcedureDataset(Dataset):
    def __init__(self, dataframe, seq_len, procedure_code):
        self.dataframe = dataframe[dataframe["codigo_procedimento"] == procedure_code].copy()  # selecionando apenas as linhas correspondentes ao código do procedimento
        self.seq_len = seq_len
        self.max_combinations = 240  # Número máximo de combinações de CNES x código de procedimento

        self.normalize_data()
        self.pad_data()

    def normalize_data(self):
        # Normalizando as colunas 'x' e 'y'
        scaler = MinMaxScaler()
        self.dataframe[['x', 'y']] = scaler.fit_transform(self.dataframe[['x', 'y']])

    def pad_data(self):
        self.dataframe['codigo_procedimento'] = self.dataframe['codigo_procedimento'].apply(
            lambda x: x + [0] * (self.max_combinations - len(x))
        )

    def __getitem__(self, index):
        x = torch.tensor(self.dataframe['codigo_procedimento'].iloc[index], dtype=torch.float32)
        y = torch.tensor(self.dataframe[['x', 'y']].iloc[index], dtype=torch.float32)
        metadata = torch.tensor(self.dataframe[['cnes', 'date']].iloc[index], dtype=torch.float32)

        return x, y, metadata

    def __len__(self):
        return len(self.dataframe) - self.seq_len
