# %%
%load_ext autoreload
%autoreload 2

from transformer_lens import HookedTransformer
import plotly.express as px
import torch
from einops import *

# from base import Config
# from utils import get_splits, Sampler
# from anthropic import AnthropicSAE
# from vanilla import VanillaSAE
# from gated import GatedSAE
# from rainbow import RainbowSAE
# %%
model = HookedTransformer.from_pretrained("gelu-1l").cuda()
# train, validation = get_splits()

# %%

config = Config(eps=0.01, lr=1e-5, whiten=True, n_buffers_for_cov = 64, n_buffers=500, expansion=4, buffer_size=2**17, sparsities=[0.1,0.2,0.4,0.6,0.8,1,2,3,4])
sampler = Sampler(config, train, model)
sae = VanillaSAE(config, model).cuda()

# %%
torch.backends.cudnn.benchmark = True
sae.train(sampler, model, validation, log=True)