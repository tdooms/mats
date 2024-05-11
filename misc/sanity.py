# %%

import torch
from einops import *
import plotly.express as px

directions = torch.randn(128, 512, device="cuda") * 2.0
mask = torch.rand_like(directions, device="cuda") > 0.95
directions = mask.float() * directions

n_instances = 4

def encode(x):
    if x.dim() == 3:
        noise = directions[torch.arange(x.size(0), device="cuda") % 128]
        print(noise.shape)
        noise = repeat(noise, "... d -> ... inst d", inst=n_instances)
    elif x.dim() == 4:
        noise = directions[torch.arange(x.size(1), device="cuda") % 128]
        noise = repeat(noise, "seq d -> batch seq inst d", batch=x.size(0), inst=n_instances)
    else:
        raise ValueError("Invalid input shape")
    print(noise.shape, x.shape)
    return x + noise


a = torch.randn((1024, 4, 512), device="cuda") * 0.0
b = encode(a)

px.imshow(b[:, 0].detach().cpu().numpy(), aspect="auto").show()