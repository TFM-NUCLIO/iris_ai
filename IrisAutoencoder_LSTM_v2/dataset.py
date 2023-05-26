import pandas as pd
import numpy as np

def create_sliding_window(data, window_size):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
    return np.array(X)

def load_and_transform_data(window_size):
    df_X2 = pd.read_pickle('E:/OneDrive/Documentos/GitHub/iris/IrisAutoencoder_LSTM_v2/train_data/df_X2.pkl')
    df_X3 = pd.read_pickle('E:/OneDrive/Documentos/GitHub/iris/IrisAutoencoder_LSTM_v2/train_data/df_X3.pkl')

    df_X2_melt = df_X2.melt(id_vars=['cnes', 'codigo_procedimento'], var_name='date', value_name='df_X2')
    df_X3_melt = df_X3.melt(id_vars=['cnes', 'codigo_procedimento'], var_name='date', value_name='df_X3')

    df = pd.merge(df_X2_melt, df_X3_melt, on=['cnes', 'codigo_procedimento', 'date'])

    grouped = df.groupby(['cnes', 'codigo_procedimento'])

    lstm_input = []
    for _, group in grouped:
        group_values = group[['df_X2', 'df_X3']].values
        X = create_sliding_window(group_values, window_size)
        lstm_input.append(X)

   # print("Número de sequências:", len(lstm_input))
   # print("Tamanho de cada sequência:", [len(seq) for seq in lstm_input])

    return lstm_input
