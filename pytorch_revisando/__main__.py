from init import *
from config import *

def load_data(data_path):
    # Carregar e processar os dados conforme necessário
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    # Função para pré-processar os dados, como transformar o formato longo para o formato wide e normalizar os dados
    # Implemente a lógica de pré-processamento aqui
    pass

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, input_dim)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = self.decoder(x)
        return x

def train_model(autoencoder, train_loader, num_epochs, learning_rate):
    loss_fn = nn.MSELoss()
    optimizer = Adam(autoencoder.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = autoencoder(batch)
            loss = loss_fn(outputs, batch)
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

def evaluate_model(autoencoder, test_df, y_true, threshold_quantile):
    test_tensor = torch.tensor(test_df.values.astype(np.float32))
    reconstructions = autoencoder(test_tensor)
    mse = np.mean(np.power(test_df.values - reconstructions.detach().numpy(), 2), axis=1)
    threshold = np.quantile(mse, threshold_quantile)

    y_pred = [1 if e > threshold else 0 for e in mse]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1 Score: {:.4f}".format(f1))

    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", conf_matrix)

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    df_wide = preprocess_data(df)

    # Divida os dados em conjuntos de treinamento e teste
    train_df, test_df, y_train, y_true = train_test_split(df_wide, y, test_size=0.3, random_state=42)

    # Normalize os dados
    scaler = StandardScaler()
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)

    # Converta os dados de treinamento para tensores e crie um DataLoader
    train_tensor = torch.tensor(train_df.astype(np.float32))
    train_dataset = torch.utils.data.TensorDataset(train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Crie o autoencoder
    input_dim = train_df.shape[1]
    autoencoder = Autoencoder(input_dim, ENCODING_DIM)

    # Treine o autoencoder
    train_model(autoencoder, train_loader, EPOCHS, LEARNING_RATE)

    # Avalie o modelo
    evaluate_model(autoencoder, test_df, y_true, THRESHOLD_QUANTILE)



