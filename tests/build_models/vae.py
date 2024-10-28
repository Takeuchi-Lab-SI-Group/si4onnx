import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

sys.path.append(os.pardir)
from si4onnx.tests.dataset.noisy_mnist import NoisyMNIST

# set seed
torch.manual_seed(0)
np.random.seed(0)

epochs = 200
batch_size = 16
lr = 0.01

# make dataloader
num_samples = 2000
train_dataset = NoisyMNIST(num_samples)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2_mean = nn.Linear(512, latent_dim)
        self.fc2_logstd = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.enc(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        mean = self.fc2_mean(x)
        logstd = self.fc2_logstd(x)
        return mean, logstd

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 64 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.Sigmoid()
        )

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), 64, 7, 7)
        x = self.dec(x)
        return x
    
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        mean, logstd = self.encoder(x)
        std = logstd
        eps = torch.randn_like(std)
        z = eps * std + mean
        recon_x = self.decoder(z)
        return recon_x, mean, logstd

    def sample(self, num_samples):
        z = torch.randn(num_samples, latent_dim).to("cpu")
        samples = self.decoder(z)
        return samples

latent_dim = 10
model = VAE(latent_dim).to("cpu")
optimizer = optim.Adam(model.parameters(), lr=0.001)

def vae_loss(recon_x, x, mean, logstd):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logstd - mean.pow(2) - logstd.exp())
    return recon_loss + kl_div
# train
for epoch in range(epochs):
    for images in train_loader:
        images = images.to("cpu")
        recon_images, mean, logstd = model(images)
        loss = vae_loss(recon_images, images, mean, logstd)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# save model
model_path = './tests/models/vae.onnx'
input_x = torch.randn(1, 1, 28, 28)
torch.onnx.export(model, input_x, model_path)

# save pth
model.eval()
pth_path = model_path.replace('.onnx', '.pth')
torch.save(model, pth_path)
