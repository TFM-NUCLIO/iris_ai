import torch
from torch import nn
from torchviz import make_dot
from graphviz import Source

# Definir o modelo
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        x, _ = self.decoder(hidden.repeat(x.size(1), 1, 1))
        return x

# Criar uma instância do modelo
input_dim = 10
hidden_dim = 5
num_layers = 2
model = LSTMAutoencoder(input_dim, hidden_dim, num_layers)

# Gerar um tensor de exemplo para visualizar o gráfico
batch_size = 1
seq_len = 5
input_tensor = torch.randn(batch_size, seq_len, input_dim)

# Calcular a saída e gerar o gráfico
output_tensor = model(input_tensor)
graph = make_dot(output_tensor, params=dict(model.named_parameters()))

# Definir a cor da fonte das palavras como negrito
graph.format = 'png'
graph.attr('node', style='filled', color='black', fontcolor='black', fontname='bold')  # Define a cor e o estilo da fonte
graph.attr('edge', color='blue')  # Define a cor das arestas

# Ajustar o tamanho da figura e salvar o arquivo de imagem
graph.render("autoencoder_diagram", format="png", cleanup=True, view=True)





