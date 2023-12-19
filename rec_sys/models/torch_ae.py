import torch
from torch import nn
from torch.nn import functional as F


class VariationalEncoder(nn.Module):
    def __init__(self, in_features=8287, latent_dims=512, device="cpu"):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(in_features, 1024)
        self.linear2 = nn.Linear(1024, latent_dims)
        self.linear3 = nn.Linear(1024, latent_dims)
        self.device = device

        self.N = torch.distributions.Normal(0, 1)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape).to(self.device)
        return z, mu, sigma


class Decoder(nn.Module):
    def __init__(self, out_features=8287, latent_dims=512, device="cpu"):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 1024)
        self.linear2 = nn.Linear(1024, out_features)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        return z


class AEModel(nn.Module):
    def __init__(self, in_out_features=8287, latent_dims=512, device="cpu"):
        super(AEModel, self).__init__()
        self.encoder = VariationalEncoder(in_out_features, latent_dims, device=device)
        self.decoder = Decoder(in_out_features, latent_dims)

    def forward(self, x):
        z, mu, sigma = self.encoder(x)
        return self.decoder(z), mu, sigma
