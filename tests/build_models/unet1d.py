import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# シード設定
torch.manual_seed(0)
np.random.seed(0)

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# カスタムデータセットの作成
class TimeSeriesDataset(Dataset):
    def __init__(self, num_samples, sequence_length=128, noise_std=0.2):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.noise_std = noise_std
        
        self.data = []
        self.labels = []
        
        for _ in range(num_samples):
            # 1つの区切り位置をランダムに生成
            segment_point = np.random.randint(20, sequence_length-20)  # 両端を避ける
            
            # クリーンな信号の生成（2つのセグメントで異なる正規分布）
            clean_signal = np.zeros(sequence_length)
            labels = np.zeros(sequence_length)
            
            # 2つのセグメントのパラメータ生成
            mean1, mean2 = np.random.normal(0, 1, 2)  # 平均値は標準正規分布から生成
            std = 1  # 標準偏差は1で固定
            
            # 最初のセグメント
            clean_signal[:segment_point] = np.random.normal(mean1, std, segment_point)
            labels[:segment_point] = 0
            
            # 2番目のセグメント
            clean_signal[segment_point:] = np.random.normal(mean2, std, sequence_length - segment_point)
            labels[segment_point:] = 1
            
            # ガウシアンノイズの追加
            noise = np.random.normal(0, self.noise_std, sequence_length)
            noisy_signal = clean_signal + noise
            
            self.data.append(noisy_signal)
            self.labels.append(labels)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx]).unsqueeze(0)  # [1, sequence_length]
        y = torch.FloatTensor(self.labels[idx])  # [sequence_length]
        return x, y

# 1D UNetのブロック定義（変更なし）
class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv1D(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# 1D UNetモデル定義（出力チャンネルを1に変更）
class UNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv1D(1, 32)
        self.down1 = Down1D(32, 64)
        self.down2 = Down1D(64, 128)
        self.up1 = Up1D(128, 64)
        self.up2 = Up1D(64, 32)
        self.outc = nn.Conv1d(32, 1, kernel_size=1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.outc(x)

# トレーニング設定
def train_model():
    epochs = 20
    batch_size = 32
    sequence_length = 128
    
    # データローダーの作成
    train_dataset = TimeSeriesDataset(epochs * batch_size, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # モデル、損失関数、オプティマイザーの設定
    model = UNet1D().to(device)
    criterion = nn.BCEWithLogitsLoss()  # 2値分類用の損失関数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # トレーニングループ
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(1), target)  # outputのチャンネル次元を削除
            loss.backward()
            optimizer.step()
            
        print(f'Epoch [{epoch+1}/{epochs}]', f'Loss: {loss.item():.4f}')
    
    return model

# モデルの保存
def save_model(model, save_path='./tests/models/unet1d.onnx'):
    model.eval()
    dummy_input = torch.randn(1, 1, 128).to(device)
    torch.onnx.export(model, dummy_input, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    model = train_model()
    save_model(model)