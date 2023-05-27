import torch

class Config:
    INPUT_DIM = 240  # Tamanho do vetor de entrada após o padding
    BATCH_SIZE = 64
    NUM_EPOCHS = 5
    LEARNING_RATE = 0.001
    HIDDEN_DIM = 50
    NUM_LAYERS = 2
    WINDOW_SIZE = 12
    THRESHOLD = 0.95
    DATAPATH = 'E:/OneDrive/Documentos/GitHub/iris/Iris_LSTM_procedimento/train_data/df_masked.pkl'
    isCuda = torch.cuda.is_available()
