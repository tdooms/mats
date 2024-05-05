# %%
from transformer_lens import HookedTransformer
import plotly.express as px
import torch

torch.set_grad_enabled(False)

neo = HookedTransformer.from_pretrained("gpt-neo-small")
gpt = HookedTransformer.from_pretrained("gpt2-small")

# subsample = torch.randperm(model.cfg.d_vocab)[:2000].to(model.cfg.device)
# W_E = model.W_E[subsample] # Take a random subset of 5,000 for memory reasons
# W_E_normed = W_E / W_E.norm(dim=-1, keepdim=True) # [d_vocab, d_model]
# cosine_sims = W_E_normed @ W_E_normed.T # [d_vocab, d_vocab]

# px.histogram(cosine_sims.flatten().detach().cpu().numpy(), title="Pairwise cosine sims of embedding")

# %%
# subsample = torch.randperm(neo.cfg.d_vocab)[:2000].to(neo.cfg.device)
# px.scatter(neo.W_E[subsample].abs().mean(1).sort().values.detach().cpu())
# px.scatter(gpt.W_E[subsample].pow(2).mean(1).sort().values.detach().cpu())

# model = neo
neo_normed = neo.W_E / neo.W_E.norm(dim=-1, keepdim=True)
gpt_normed = gpt.W_E / gpt.W_E.norm(dim=-1, keepdim=True)

distances = torch.stack([neo_normed, gpt_normed], dim=0).mean(1).abs()

px.scatter(means.sort(-1, descending=True).values.T.cpu(), title="Mean embedding values sorted").show()

print(f"neo: deviation from 0  {means[0].abs().sum().item():.2f}")
print(f"gpt: deviation from 0  {means[1].abs().sum().item():.2f}")

# W_E_normed.mean(0).abs().sum()