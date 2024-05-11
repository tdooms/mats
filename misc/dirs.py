# %%
%load_ext autoreload
%autoreload 2

from misc.cp import AnthropicSAE, Config, Sampler, get_splits
from transformer_lens import HookedTransformer
from einops import *
import plotly.express as px
import torch
from transformer_lens import utils
# %%
training, validation = get_splits(32)
model = HookedTransformer.from_pretrained("gelu-1l")
# %%

config = Config(sparsities=(0.1, 0.2, 0.5, 1.0))
sampler = Sampler(config, training, model)
# sae = AnthropicSAE.from_pretrained("anthropic-sae.pt", config=config, model=model)
sae  = AnthropicSAE(config, model).cuda()

# %%

sae.train(sampler, model, validation, log=True)

# %%

torch.save(sae.state_dict(), "sae-noise-0.3.pt")
# %%

hook_pt = utils.get_act_name(config.point, config.layer)
loss, cache = model.run_with_cache("The first day", names_filter=[hook_pt])

act = cache[hook_pt][0, -2:-1]
# %%

x = repeat(act, "... d -> ... inst d", inst=4)
# torch.zeros(1, 4, 512).cuda()
# px.imshow(sae.noise(x)[0, 0].cpu().view(32, -1), aspect="auto")
input = repeat(act, "... d -> ... inst d", inst=4)

output = sae.forward(input)

px.imshow(output[0, 0].view(32, -1).detach().cpu(), color_continuous_midpoint=0.0, aspect="auto", color_continuous_scale="RdBu")

# diff =  output[:, 0] - input[:, 0]
# px.imshow(input[0, 0].cpu().view(32, 16), color_continuous_midpoint=0.0, aspect="auto", color_continuous_scale="RdBu")
# %%
from scipy.spatial.distance import cdist
a = [1 - torch.from_numpy(cdist(sae.W_dec[i].detach().cpu().numpy(), sae.directions.detach().cpu().numpy(), metric="cosine")) for i in range(4)]
a = torch.stack(a)
# px.imshow(a.T, color_continuous_midpoint=0, color_continuous_scale="RdBu", aspect="auto")

a.max(1).values.mean(-1)

# px.line((a > 0.5).sum(0).sort().values)
# %%
