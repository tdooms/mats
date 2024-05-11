# %%
import torch
from einops import *

z = torch.randn(12, 64)

enc = torch.randn(64, 512)
dec = torch.randn(512, 64)

# dynamic_gate = torch.randn(64, 12, 512)
# gate = einsum(dynamic_gate, z, "d_head n_head d_sae, n_head d_head -> n_head d_sae").relu()

gate = torch.randn(12, 512)

# gate_proto = torch.randn

hid = einsum(z, enc, gate, "n_head d_head, d_head d_sae, n_head d_sae -> d_sae")
act = hid.relu()
out = einsum(act, gate, dec, "d_sae, n_head d_sae, d_head d_sae -> n_head d_head")

# %%
from base import Loss, Config, BaseSAE
from torch import nn

class AttnSAE(BaseSAE):
    def __init__(self, config, model) -> None:
        super().__init__(config, model)
        device = config.device
        self.n_heads = model.cfg.n_heads
        
        W_dec = torch.randn(self.n_instances, self.d_model, self.n_heads, self.d_hidden, device=device)
        W_dec /= torch.norm(W_dec, dim=-2, keepdim=True)
        
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.mT.clone())
        
        W_gate = torch.randn(self.n_instances, self.n_heads, self.n_heads, self.d_model, device=device)
        torch.nn.init.kaiming_normal_(W_gate)
        self.W_gate = nn.Parameter(W_gate)
        
        self.b_dec = nn.Parameter(torch.zeros(self.n_instances, self.d_model, device=device))
        self.b_gate = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))
        self.b_enc = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))
    
    def encode(self, x):
        centered = x - self.b_dec # [batch inst n_head d_model]
        
        heads = einsum(centered, self.W_enc, "... inst n_head d_head, inst d_head d_hidden -> ... inst n_head d_hidden") + self.b_enc
        gate = einsum(centered, self.W_gate, "... inst n_head, inst d_hidden n_head -> ... inst n_head") + self.b_gate
        
        return einsum(heads, gate.relu(), "... inst n_head d_hidden, ... inst n_head -> ... inst d_hidden")
    
    def decode(self, x):
        # return einsum(x, self.W_dec, "... inst d_hidden, inst d_hidden d_head -> ... inst d_head") + self.b_dec
        
        
        
        

config = Config()