import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import si4onnx.utils as utils
import si4onnx.datasets as 

sys.path.append(os.pardir)

utils.set_thread()
utils.set_seed(0)


# set seed
torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cpu')

epochs = 10
batch_size = 16

# make dataloader
num_samples = 1000

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# define model
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.up1 = Up(128, 64)
        self.up2 = Up(64, 32)
        self.outc = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        output = self.outc(x)
        return output

model = UNet().to(device)

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
model_path = './tests/models/unet.onnx'
input_x = torch.randn(1, 1, 28, 28)
torch.onnx.export(model, input_x, model_path)

# save pth
model.eval()
pth_path = model_path.replace('.onnx', '.pth')
torch.save(model, pth_path)