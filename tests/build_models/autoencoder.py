import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.pardir)
from si4onnx.tests.dataset.noisy_mnist import NoisyMNIST

# set seed
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 100
batch_size = 32
lr = 0.01

# make dataloader
num_samples = epochs * batch_size
train_dataset = NoisyMNIST(num_samples)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# define model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 4, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = Autoencoder()
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# train
for epoch in range(epochs):
   for images in train_loader:

       outputs = model(images)
       loss = criterion(outputs, images)
       
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       
   print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# save model
model_path = './tests/models/autoencoder.onnx'
input_x = torch.randn(1, 1, 28, 28)
torch.onnx.export(model, input_x, model_path)