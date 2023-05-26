import pandas as pd
import numpy as np

def load_and_transform_data():
    df_X1 = pd.read_pickle('./train_data/df_X1.pkl')
    df_X2 = pd.read_pickle('./train_data/df_X2.pkl')
    df_X3 = pd.read_pickle('./train_data/df_X3.pkl')

    # Resetando os índices para transformá-los em colunas
    df_X1.reset_index(inplace=True)
    df_X2.reset_index(inplace=True)
    df_X3.reset_index(inplace=True)

    # "Derretendo" os dataframes e atribuindo nomes de variáveis correspondentes
    df_X1_melt = df_X1.melt(id_vars=['cnes', 'codigo_procedimento'], var_name='date', value_name='df_X1')
    df_X2_melt = df_X2.melt(id_vars=['cnes', 'codigo_procedimento'], var_name='date', value_name='df_X2')
    df_X3_melt = df_X3.melt(id_vars=['cnes', 'codigo_procedimento'], var_name='date', value_name='df_X3')

    # Fundindo os dataframes derretidos com base em cnes, codigo_procedimento e data
    df = pd.merge(df_X1_melt, df_X2_melt, on=['cnes', 'codigo_procedimento', 'date'])
    df = pd.merge(df, df_X3_melt, on=['cnes', 'codigo_procedimento', 'date'])

    # Ordenando por cnes, codigo_procedimento e date
    df = df.sort_values(by=['cnes', 'codigo_procedimento', 'date'])

    # Convertendo para o formato apropriado para LSTM ([samples, timesteps, features])
    grouped = df.groupby(['cnes', 'codigo_procedimento'])

    lstm_input = []
    for _, group in grouped:
        group_values = group[['df_X1', 'df_X2', 'df_X3']].values  # Extrai apenas os valores dos dataframes
        lstm_input.append(group_values)

    lstm_input = np.array(lstm_input)  # Converte para um numpy array

    print("Tamanho do lstm_input:", lstm_input.shape)

    """if lstm_input.shape[0] > 0:
        print("Exemplo de um objeto lstm_input:")
        print(lstm_input[0])
"""
    return lstm_input