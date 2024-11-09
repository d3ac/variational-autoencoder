from torch import nn
import torch
import numpy as np
import torch.nn.functional as F

def log_normal_p(x, mu, std):
    return -0.5 * np.log(2 * np.pi) - torch.log(std) - 0.5 * (x - mu) ** 2 / std ** 2

class Encoder(nn.Module):
    """
    encoder的目的是将输入数据x映射到潜在空间中的分布, 也就是学习到 x -> z的映射,
    当然我们也可以不学习, 直接假设z是一个标准正态分布, 这样我们就不需要encoder了, 这种就是VAE的假设,
    在整个训练完了之后, 我们可以直接从标准正态分布中采样z, 然后通过decoder生成数据,
    如何计算p(x)呢, 可以通过p(x) = sum[p(x|z)p(z)]来计算, 当然这个计算是不可行的, 因为z是连续的,
    所以我们只会采样一些z, 然后计算p(x|z)p(z), 这样就可以近似计算p(x)了, 这一些z是通过encoder得到的,
    也就是把这里的p(z)换成p(z|x), 用一个x生成很多的z(encode), 再把这个z拿去生成很多的x^hat(decode), 然后求mean得到p(x)
    """
    def __init__(self, latent_dim, data_dim):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 2 * latent_dim), nn.ReLU(),
            nn.Linear(2 * latent_dim, 2 * latent_dim), nn.ReLU(),
            nn.Linear(2 * latent_dim, 2 * latent_dim)
        )
    
    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std
    
    def encode(self, x):
        mu, log_var = torch.chunk(self.net(x), 2, dim=-1)
        return mu, log_var
    
    def log_prob(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterization(mu, log_var)
        return log_normal_p(z, mu, torch.exp(0.5 * log_var)).sum(dim=-1)
    
    def forward(self, x):
        return self.log_prob(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 2 * data_dim), nn.ReLU(),
            nn.Linear(2 * data_dim, 2 * data_dim), nn.ReLU(),
            nn.Linear(2 * data_dim, data_dim)
        )
    
    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        # Encode
        mu, log_var = self.encoder.encode(x)
        z = self.encoder.reparameterization(mu, log_var)
        # ELBO
        reconstruction_loss = F.mse_loss(self.decoder(z), x) # 越小越好
        KL_divergence = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1), dim=0) # 因为假设了p(z)是标准正态分布, q(z|x)是(mu, std)的正态分布, 所以化简后就是这样了
        loss = reconstruction_loss #+ 0.00025 * KL_divergence
        return loss
    
    def reconstruct(self, x, dim, dataset):
        mu, log_var = self.encoder.encode(x)
        z = self.encoder.reparameterization(mu, log_var)
        if dataset == "MNIST":
            return self.decoder(z).reshape(-1, 1, dim, dim)
        else:
            return self.decoder(z).reshape(-1, 3, dim, dim)