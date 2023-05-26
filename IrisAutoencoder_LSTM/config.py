# Parâmetros do modelo
INPUT_DIM = None  # Será definido com base no conjunto de dados
HIDDEN_DIM = 50
NUM_LAYERS = 2
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
BATCH_SIZE = 32
THRESHOLD = 0.01  # Limiar para detecção de anomalias

# Parâmetros dos dados
DATA_PATHS = {
    'df_X1': 'train_data/df_X1.pkl',
    'df_X2': 'train_data/df_X2.pkl',
    'df_X3': 'train_data/df_X3.pkl'
}

