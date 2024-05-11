import torch
from torch import nn
from einops import *
from torch.optim import Adam

from base import BaseSAE, Loss


class AnthropicSAE(BaseSAE):
    def __init__(self, config, model):
        super().__init__(config, model)
        device = config.device

        W_dec = torch.randn(self.n_instances, self.d_hidden, self.d_model, device=device)
        W_dec /= torch.norm(W_dec, dim=-1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)

        W_enc = W_dec.mT.clone().to(device)
        self.W_enc = nn.Parameter(W_enc)

        self.b_enc = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))
        self.b_dec = nn.Parameter(torch.zeros(self.n_instances, self.d_model, device=device))

        self.relu = nn.ReLU()
        self.optimizer = Adam(self.parameters(), lr=self.config.lr, betas=(0.9, 0.999))

    def encode(self, x):
        return self.relu(einsum(x - self.b_dec, self.W_enc, "... inst d, inst d hidden -> ... inst hidden") + self.b_enc),

    def decode(self, h):
        return einsum(h, self.W_dec, "... inst hidden, inst hidden d -> ... inst d") + self.b_dec

    def loss(self, x, x_hid, x_hat, fraction):
        reconstruction = ((x_hat - x) ** 2).mean(0).sum(dim=-1)

        norm = self.W_dec.norm(dim=-1)
        lambda_ = min(1, fraction * 20)
        
        sparsity = einsum(x_hid, norm, "batch inst hidden, inst hidden -> batch inst").mean(dim=0)

        return Loss(reconstruction, lambda_ * sparsity, torch.zeros(self.n_instances, device=x.device))
