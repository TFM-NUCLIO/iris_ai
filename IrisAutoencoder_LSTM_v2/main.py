import torch
from torch.utils.data import DataLoader
from dataset import load_and_transform_data
from model import LSTMAutoencoder
import config
import sys

def confirm(prompt="Confirmar a continuação (s/n): "):
    while True:
        answer = input(prompt).lower()
        if answer == 's':
            return True
        elif answer == 'n':
            print("Processo interrompido pelo usuário.")
            sys.exit(0)
        else:
            print("Resposta inválida. Responda com 's' para sim ou 'n' para não.")

def main():
    if confirm("Deseja prosseguir para o treinamento?"):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache()

        datasets = load_and_transform_data(config.WINDOW_SIZE)

        models = [LSTMAutoencoder(2, config.HIDDEN_DIM, config.NUM_LAYERS).to(DEVICE) for _ in datasets]

        for model, dataset in zip(models, datasets):
            dataset = torch.Tensor(dataset).to(DEVICE)
            model.train_model(dataset, DEVICE, config.BATCH_SIZE, config.LEARNING_RATE, config.NUM_EPOCHS, config.WINDOW_SIZE * 12)
            model.evaluate(dataset, DEVICE)
            anomalies = model.detect_anomalies(dataset, DEVICE, config.THRESHOLD)
            print(anomalies)
    else:
        print("Processo interrompido pelo usuário.")
        sys.exit(0)

if __name__ == "__main__":
    main()
