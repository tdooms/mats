# %%
%load_ext autoreload
%autoreload 2

from transformer_lens import HookedTransformer, patching, HookedSAE, HookedSAEConfig
from einops import *
import torch
import plotly.express as px
from transformer_lens import utils
import itertools
import pandas as pd
from torch import nn
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from transformer_lens.utils import download_file_from_hf

# %%
layer = 9
hook_pt = utils.get_act_name("resid_pre", layer)

hf_repo = "jbloom/GPT2-Small-SAEs-Reformatted" 
cfg_hf = download_file_from_hf(hf_repo, f"{hook_pt}/cfg.json")

def res_sae_cfg_to_hooked_sae_cfg(sae_cfg):
    new_cfg = {
        "d_sae": sae_cfg["d_sae"],
        "d_in": sae_cfg["d_in"],
        "hook_name": sae_cfg["hook_point"],
    }
    return HookedSAEConfig.from_dict(new_cfg)

cfg = res_sae_cfg_to_hooked_sae_cfg(cfg_hf)
state_dict = download_file_from_hf(hf_repo, file_name=f"{hook_pt}/sae_weights.safetensors")

sae = HookedSAE(cfg)
sae.load_state_dict(state_dict)

# %%
model = HookedTransformer.from_pretrained("gpt2-small")

# %%
def get_logit_diff(logits, answer_token_indices):
    answer_token_indices = answer_token_indices.to(logits.device)
    
    if len(logits.shape) == 3:
        logits = logits[:, -1, :]
        
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits)

template = [
    "When{} and{} went to the shops.{} gave the bag to",
    "When{} and{} went to the park.{} gave the ball to",
    "When{} and{} went to the shops.{} gave an apple to",
    "When{} and{} went to the park.{} gave a drink to",
]

names = [
    (" John", " Mary"),
    (" Tom", " James"),
    (" Dan", " Mark"),
    (" Martin", " Amy"),
    (" Joseph", " Will"),
    (" Bob", " Jack")
]

prompts, answers = [], []
for prompt, name in itertools.product(template, names):
    # prompts += [prompt.format(name[0], name[1], name[0]), prompt.format(name[1], name[0], name[1])] 
    # answers += [(name[1], name[0]), (name[0], name[1])]
    prompts += [prompt.format(name[0], name[1], name[0])]
    answers += [(name[1], name[0])]

pd.options.display.max_colwidth = 100
pd.DataFrame(prompts, answers)
# %%
    
model.reset_hooks(including_permanent=True)
b_noise = torch.randn(1, 24576, device="cuda") * 0.0001
b_noise = nn.Parameter(b_noise)

for param in model.parameters():
    param.requires_grad = False


def noise_hook(acts, hook):
    acts[:, -1, :] += b_noise.relu() @ sae.W_dec
    return acts

hook_pt = utils.get_act_name("resid_pre", layer)

optimizer = torch.optim.Adam([b_noise], lr=0.005)

answer_idxs = torch.tensor([[model.to_single_token(name) for name in pair] for pair in answers])

original_logits = model(prompts, return_type="logits").detach()
original_logits = original_logits[:, -1, :]

mask = torch.ones_like(original_logits)

for i in range(mask.size(0)):
    mask[i, answer_idxs[i, 0]] = 0
    mask[i, answer_idxs[i, 1]] = 0


og_logit_diffs = get_logit_diff(original_logits, answer_idxs)
alpha = 5
beta = 0.02

# %%
loss_metrics = defaultdict(list)
model.add_perma_hook(hook_pt, noise_hook)

for i in tqdm(range(1000)):
    logits = model.run_with_hooks(prompts, return_type="logits")
    logit_diffs = get_logit_diff(logits, answer_idxs)
    
    diff_loss = logit_diffs.mean()
    
    # diff_loss = (logit_diffs + og_logit_diffs).pow(2).mean()
    
    diffs = (logits[:, -1, :] - original_logits)
    diffs = diffs * mask
    
    retain_loss = diffs.pow(2).mean()
    norm_loss = beta * b_noise.abs().sum()
    loss = diff_loss + alpha * retain_loss + norm_loss
    
    # norm_loss = beta * b_noise.abs().sum()
    # loss = diff_loss + retain_loss

    loss_metrics['diff_loss'].append(diff_loss.item())
    loss_metrics['retain_loss'].append(retain_loss.item())
    loss_metrics['norm_loss'].append(norm_loss.item())
    loss_metrics['total_loss'].append(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

metrics_pd = pd.DataFrame.from_dict(loss_metrics)

px.line(
    metrics_pd,
    y=["diff_loss", "total_loss", "retain_loss", "norm_loss"],
    x=np.arange(len(loss_metrics['diff_loss'])),
).show()

# %%

model.add_perma_hook(hook_pt, noise_hook)
# logits = model.run_with_hooks("When Sid and Trump went to the shops. Trump gave the bag to", return_type="logits")
logits = model.run_with_hooks("When John and Mary went to the shops. John gave the bag to", return_type="logits")

logit_values, logit_idxs = logits[:, -1, :].topk(5, dim=-1)
logit_toks = [model.to_str_tokens(x) for x in logit_idxs]

print(logit_toks, logit_values)

# %%

# px.imshow(b_noise[0].view(-1, 256).detach().cpu()).show()

print("L0", (b_noise > 0.01).sum(), "L1", b_noise.abs().sum())

# vals, idxs = b_noise[0].sort()
# px.line(y=vals.detach().cpu(), x=[f"{i}" for i in idxs]).show()