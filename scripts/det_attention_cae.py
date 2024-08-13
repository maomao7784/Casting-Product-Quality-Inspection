import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.fc = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Linear projections
        query = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, -1)
        key = self.key_conv(x).view(batch_size, self.num_heads, self.head_dim, -1)
        value = self.value_conv(x).view(batch_size, self.num_heads, self.head_dim, -1)
        
        # Scaled dot-product attention
        energy = torch.einsum("bnqd,bnkd->bnqk", [query, key]) / (self.head_dim ** 0.5)
        attention = F.softmax(energy, dim=-1)
        out = torch.einsum("bnqk,bnvd->bnqd", [attention, value])
        
        # Concatenate heads
        out = out.view(batch_size, -1, width * height)
        out = out.view(batch_size, C, width, height)
        
        out = self.fc(out)
        out = self.gamma * out + x
        return out

class AttentionAutoencoder(nn.Module):
    def __init__(self):
        super(AttentionAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7)
        )

        # Multi-Head Attention Layer
        self.attention = MultiHeadAttention(64, 64, num_heads=8)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.attention(x)  # Apply multi-head attention here
        x = self.decoder(x)
        return x