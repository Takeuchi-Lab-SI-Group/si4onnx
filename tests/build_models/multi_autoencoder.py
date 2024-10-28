import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from tests.dataset.multi_noisy_mnist import MultiNoisyMNIST

# set seed
torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiInputMultiOutputAutoencoder(nn.Module):
    def __init__(self, input_shapes, hidden_dim, latent_dim):
        super().__init__()
        self.input_shapes = input_shapes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(np.prod(input_shape), hidden_dim),
                nn.ReLU()
            ) for input_shape in input_shapes
        ])
        
        self.encoder_combined = nn.Sequential(
            nn.Linear(hidden_dim * len(input_shapes), latent_dim),
            nn.ReLU()
        )
        
        self.decoder_combined = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * len(input_shapes)),
            nn.ReLU()
        )
        
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, np.prod(input_shape)),
                nn.ReLU()
            ) for input_shape in input_shapes
        ])
    
    def forward(self, *inputs):
        encoded = []
        for i, input_data in enumerate(inputs):
            input_data = input_data.view(input_data.size(0), -1)
            encoded_input = self.encoders[i](input_data)
            encoded.append(encoded_input)
        
        encoded_combined = torch.cat(encoded, dim=1)
        latent = self.encoder_combined(encoded_combined)
        
        decoded_combined = self.decoder_combined(latent)
        decoded_split = torch.split(decoded_combined, self.hidden_dim, dim=1)
        
        reconstructed = []
        for i, decoded in enumerate(decoded_split):
            reconstructed_input = self.decoders[i](decoded)
            reconstructed_input = reconstructed_input.view(inputs[i].size())
            reconstructed.append(reconstructed_input)
        
        return reconstructed

# Hyperparameters
num_samples = 1000
input_shapes = [(1, 28, 28), (1, 28, 28), (1, 28, 28)]
hidden_dim = 128
latent_dim = 64
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Create dataset and dataloader
dataset = MultiNoisyMNIST(num_samples, len(input_shapes))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create model, loss function, and optimizer
model = MultiInputMultiOutputAutoencoder(input_shapes, hidden_dim, latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for data in dataloader:
        inputs = [data[:, i] for i in range(len(input_shapes))]
        
        # Forward pass
        outputs = model(*inputs)
        loss = sum([criterion(output, input_data) for output, input_data in zip(outputs, inputs)])
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
# inference by torch (Debug)
dummy_inputs = [torch.randn(1, *input_shape) for input_shape in input_shapes]
output_data = model(dummy_inputs[0], dummy_inputs[1], dummy_inputs[2])
print("len(ouput_data) by forward on pytorch:", len(output_data))

# Save model
model_path = './tests/models/multi_autoencoder.onnx'
dummy_inputs = tuple(torch.randn(1, *input_shape) for input_shape in input_shapes) # onnx model requires tuple
torch.onnx.export(model, dummy_inputs, model_path, input_names=['input_{}'.format(i) for i in range(len(input_shapes))], output_names=['output_{}'.format(i) for i in range(len(input_shapes))])
