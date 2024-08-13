import torch
import torch.nn as nn

# The loss function of a VAE (Variational Autoencoder) consists of two parts: 
# the reconstruction loss and the KL divergence (Kullback-Leibler Divergence).
# https://avandekleut.github.io/vae/
# https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        # Input: (1, 224, 224)
        # Encoder
        super(VariationalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # (16, 112, 112)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (32, 56, 56)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7) # (64, 50, 50)
        )

        self.flatten = nn.Flatten()
        self.norm = torch.distributions.Normal(0, 1)
        self.fc_mean = nn.Linear(64*50*50, latent_dim) # mean value layer
        self.fc_sigma = nn.Linear(64*50*50, latent_dim) # std value layer
        # tracking the KL divergence
        self.kl = 0

    def forward(self, x):
        # encoder
        x = self.encoder(x)
        x = self.flatten(x)
        mean = self.fc_mean(x)
        sigma = torch.exp(self.fc_sigma(x))
        z = mean + sigma*self.norm.sample(mean.shape).cuda()
        self.kl = (sigma**2 + mean**2 - torch.log(sigma) - 0.5).sum()
        
        return z

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 64*50*50)  # map latent space to encoder's output shape
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        z = self.encoder(x)
        z = self.fc_dec(z)
        z = z.view(-1, 64, 50, 50)  # reshape to match encoder's last conv output
        z = self.decoder(z)
        return z

