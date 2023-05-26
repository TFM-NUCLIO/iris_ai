import torch
from torch.utils.data import DataLoader
from dataset import load_and_transform_data
from model import LSTMAutoencoder
import config

def main():
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

    # Criar os conjuntos de dados
    datasets = load_and_transform_data()

    # Criar os modelos
    models = [LSTMAutoencoder(dataset.shape[1], config.HIDDEN_DIM, config.NUM_LAYERS).to(DEVICE) for dataset in datasets]

    for model, dataset in zip(models, datasets):
        # Converter o conjunto de dados para torch.Tensor
        dataset = torch.Tensor(dataset).to(DEVICE)

        # Treinar o modelo
        model.train_model(dataset, DEVICE, config.BATCH_SIZE, config.LEARNING_RATE, config.NUM_EPOCHS)

        # Avaliar o modelo
        model.evaluate(dataset, DEVICE)

        # Detectar anomalias
        anomalies = model.detect_anomalies(dataset, DEVICE, config.THRESHOLD)
        print(anomalies)
        # ...imprimir / salvar essas anomalias 
        
if __name__ == "__main__":
    main()