import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class ProcedureDataset(Dataset):
    def __init__(self, dataframe, seq_len, procedure_code):
        self.dataframe = dataframe[dataframe["codigo_procedimento"] == procedure_code].copy()  # selecionando apenas as linhas correspondentes ao código do procedimento
        self.seq_len = seq_len
        self.normalize_data()

    def normalize_data(self):
        # Normalizando as colunas 'x' e 'y'
        self.dataframe['x'] = (self.dataframe['x'] - self.dataframe['x'].min()) / (self.dataframe['x'].max() - self.dataframe['x'].min())
        self.dataframe['y'] = (self.dataframe['y'] - self.dataframe['y'].min()) / (self.dataframe['y'].max() - self.dataframe['y'].min())
        self.data = torch.tensor(self.dataframe[['x', 'y']].values, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(self.dataframe[['x', 'y']].values, dtype=torch.float32)

    def __getitem__(self, index):
        x = self.data[index:index+self.seq_len]
        y = self.data[index+self.seq_len:index+self.seq_len+1, -1]
        
        # Retornando também os identificadores 'cnes', 'codigo_procedimento' e 'date'
        meta_info = self.dataframe.iloc[index+self.seq_len:index+self.seq_len+1][['cnes', 'codigo_procedimento', 'date']]
        
        return x, y, meta_info.to_dict('list')

    def __len__(self):
        return len(self.dataframe) - self.seq_len - 1
