import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Caminhos dos dados, se aplicável
DATA_PATH = "path/to/your/data"

def preprocess_data(df):
    # Transformar o formato longo para o formato wide
    df_wide = df.pivot_table(index=['CNES', 'MES_ANO'], columns='CODIGO_PROCEDIMENTO', values='Valor dos procedimentos por tipo de procedimento e por estabelecimento por mes/ano').reset_index()

    # Adicionar colunas adicionais
    df_wide['quantidade_procedimentos_10k_habitantes'] = df['Quantidade de procedimentos por estabelecimento por 10 mil habitantes por mes/ano']
    df_wide['valor_total_procedimentos_10k_habitantes'] = df['Valor total de procedimentos do item 12 por estabelecimento por 10 mil habitantes or mes/ano']
    df_wide['distancia_quantidade_total_procedimentos'] = df['distância da quantidade total de procedimentos por estabelecimento por mes/ano da mediana120 meses']
    df_wide['distancia_valor_total_procedimentos'] = df['distância dos valores totais dos procesimentos por estabelecimento da mediana dos 120 meses']

    # Padronizar/normalizar os dados
    scaler = StandardScaler()
    df_wide[df_wide.columns[2:]] = scaler.fit_transform(df_wide[df_wide.columns[2:]])

    return df_wide

"""Esta função primeiro chama a função pivot_table() do pandas para converter os dados de formato longo para formato wide, onde cada linha representa um estabelecimento de saúde em um determinado mês/ano e cada coluna representa o valor dos procedimentos de um determinado tipo de procedimento. Em seguida, adiciona as colunas adicionais especificadas na pergunta e normaliza/padroniza os dados usando a classe StandardScaler() do scikit-learn."""
