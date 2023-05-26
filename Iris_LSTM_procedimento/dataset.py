import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class ProcedureDataset(Dataset):
    def __init__(self, data, window_size):
        scaler = MinMaxScaler()
        self.data = torch.FloatTensor(scaler.fit_transform(data))
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        return (self.data[idx:idx+self.window_size],
                self.data[idx:idx+self.window_size])
