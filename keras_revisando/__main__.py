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

def create_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="relu")(input_layer)
    decoder = Dense(input_dim, activation="linear")(encoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    return autoencoder

def train_model(autoencoder, train_df):
    autoencoder.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mean_squared_error')
    autoencoder.fit(train_df, train_df, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_split=0.1)

def evaluate_model(autoencoder, test_df, y_true, threshold_quantile):
    reconstructions = autoencoder.predict(test_df)
    mse = np.mean(np.power(test_df - reconstructions, 2), axis=1)
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

    # Crie o autoencoder
    input_dim = train_df.shape[1]
    autoencoder = create_autoencoder(input_dim, ENCODING_DIM)

    # Treine o autoencoder
    train_model(autoencoder, train_df)

    # Avalie
    evaluate_model(autoencoder, test_df, y_true, THRESHOLD_QUANTILE)


