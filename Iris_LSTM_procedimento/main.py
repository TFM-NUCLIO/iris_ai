import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from dataset import ProcedureDataset
from model import LSTMAutoencoder
from config import Config

def main():
    # Configurações iniciais
    config = Config()

    # Carregando o dataframe
    df = pd.read_pickle(config.DATAPATH)

    def get_sorted_unique_procedures(df):
        return sorted(df['codigo_procedimento'].unique())

    # Lista de todos os códigos de procedimento
    procedure_codes = get_sorted_unique_procedures(df)

    # Criamos e treinamos um modelo para cada procedimento
    for procedure_code in procedure_codes:
        print(f"Treinando o modelo para o procedimento {procedure_code}")

        # Criando o dataset e o dataloader
        dataset = ProcedureDataset(df, config.WINDOW_SIZE, procedure_code)
        dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        try:
            for i, (x, _, _) in enumerate(dataloader):
                if torch.isnan(x).any():
                    print(f"NaN encontrado em batch {i + 1}")
                if torch.isinf(x).any():
                    print(f"Inf encontrado em batch {i + 1}")
                if x.shape[0] != config.BATCH_SIZE:
                    print(f"Batch {i + 1} com tamanho inválido: {x.shape[0]}")
                if x.size(0) != config.BATCH_SIZE:
                    print(f"Batch {i} com tamanho inválido: {x.size(0)}")
                continue
        except Exception as e:
            print("Ocorreu uma exceção:", str(e))

        # Criando o modelo
        model = LSTMAutoencoder(config.INPUT_DIM, config.HIDDEN_DIM, config.NUM_LAYERS, config.isCuda)

        if config.isCuda:
            model = model.cuda()

        # Definindo a função de perda e o otimizador
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        # Treinando o modelo
        for epoch in range(config.NUM_EPOCHS):
            for i, (x, _, _) in enumerate(dataloader):
                if config.isCuda:
                    x = x.cuda()

                # Ver o head do tensor de entrada
                print(f"Input tensor at epoch {epoch + 1}, batch {i + 1}:\n", x[0], "\n")
                print("Shape of x:", x.shape)

                # Forward pass
                outputs = model(x)
                outputs = outputs.view(-1, config.WINDOW_SIZE, config.INPUT_DIM)  # Redimensionar para corresponder a x
                loss = criterion(outputs, x)  # Comparando a saída com os dados de entrada originais

                # Backward e otimização
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{config.NUM_EPOCHS}], Loss: {loss.item():.4f}')

        print("Treinamento concluído com sucesso!")

if __name__ == '__main__':
    main()
