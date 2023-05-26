import torch
class Config:
    INPUT_DIM = 2 
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    HIDDEN_DIM = 50
    NUM_LAYERS = 2
    WINDOW_SIZE = 12
    THRESHOLD = 0.95
    DATAPATH = 'E:/OneDrive/Documentos/GitHub/iris/Iris_LSTM_procedimento/train_data/df.pkl'
    isCuda = torch.cuda.is_available()



