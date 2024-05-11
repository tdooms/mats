# %%
%load_ext autoreload
%autoreload 2

from transformer_lens import HookedTransformer, patching
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
b_noise = torch.randn(1, 768, device="cuda") * 0.01
b_noise = nn.Parameter(b_noise)

for param in model.parameters():
    param.requires_grad = False


def noise_hook(acts, hook):
    acts[:, -1, :] += b_noise
    return acts

hook_pt = utils.get_act_name("resid_pre", 0)

optimizer = torch.optim.Adam([b_noise], lr=0.001)

answer_idxs = torch.tensor([[model.to_single_token(name) for name in pair] for pair in answers])

original_logits = model(prompts, return_type="logits").detach()
original_logits = original_logits[:, -1, :]

mask = torch.ones_like(original_logits)

for i in range(mask.size(0)):
    mask[i, answer_idxs[i, 0]] = 0
    mask[i, answer_idxs[i, 1]] = 0


og_logit_diffs = get_logit_diff(original_logits, answer_idxs)
alpha = 3
beta = 0.01

# %%
loss_metrics = defaultdict(list)
model.add_perma_hook(hook_pt, noise_hook)

for i in tqdm(range(200)):
    logits = model.run_with_hooks(prompts, return_type="logits")
    logit_diffs = get_logit_diff(logits, answer_idxs)
    
    diff_loss = logit_diffs.mean()
    
    # diff_loss = (logit_diffs + og_logit_diffs).pow(2).mean()
    
    diffs = (logits[:, -1, :] - original_logits)
    diffs = diffs * mask
    
    retain_loss = diffs.pow(2).mean()
    loss = diff_loss + alpha * retain_loss
    
    # norm_loss = beta * b_noise.abs().sum()
    loss = diff_loss + retain_loss

    loss_metrics['diff_loss'].append(diff_loss.item())
    loss_metrics['retain_loss'].append(retain_loss.item())
    # loss_metrics['norm_loss'].append(norm_loss.item())
    loss_metrics['total_loss'].append(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

metrics_pd = pd.DataFrame.from_dict(loss_metrics)

px.line(
    metrics_pd,
    y=["diff_loss", "total_loss", "retain_loss"],
    x=np.arange(len(loss_metrics['diff_loss'])),
).show()

# logits = model.run_with_hooks(prompts, return_type="logits")
# loss = get_logit_diff(logits, answer_idxs).mean()

# %%

model.add_perma_hook(hook_pt, noise_hook)
# logits = model.run_with_hooks("When Sid and Trump went to the shops. Trump gave the bag to", return_type="logits")
logits = model.run_with_hooks("When John and Mary went to the shops. John gave the bag to", return_type="logits")

logit_values, logit_idxs = logits[:, -1, :].topk(5, dim=-1)
logit_toks = [model.to_str_tokens(x) for x in logit_idxs]

print(logit_toks, logit_values)

# clean_answer_logits = logits.gather(-1, answer_token_indices[:,0].unsqueeze(1)).squeeze()

# utils.test_prompt("When Jeff and Victor went to the pool, Jeff gave the ball to", " Victor", model)
# utils.test_prompt("When John and Mary went to the park. John gave the ball to", " Mary", model)
# px.line(get_logit_diff(logits,answer_idxs)[:, 0].detach().cpu())

# %%
# answer_idxs
# logits_no_names = torch.cat([logits[:,-1,0:], logits[:,-1,i+1:]])

layer = 8
head = 10

prompt = "When Mary and John went to the shops. Mary gave the bag to"
attn_pt = utils.get_act_name("attn", layer)

model.add_perma_hook(hook_pt, noise_hook)
_, corrupt_cache = model.run_with_cache(prompt, return_type="logits", names_filter=[attn_pt])

model.reset_hooks(including_permanent=True)
_, clean_cache = model.run_with_cache(prompt, return_type="logits", names_filter=[attn_pt])

clean_pattern = clean_cache[f'blocks.{layer}.attn.hook_pattern'][0, head]
corrupt_pattern = corrupt_cache[f'blocks.{layer}.attn.hook_pattern'][0, head]

empty = "\u200B"
toks = [f'{t}{empty * i}' for i, t in enumerate(model.to_str_tokens(prompt))]

patterns = torch.stack([clean_pattern, corrupt_pattern]).detach().cpu()
px.imshow(patterns, x=toks, y=toks, facet_col=0).show()
# pattern.shape

# %%

prompt = "When Sid and Trump went to the shops. Trump gave the bag to"
model.add_perma_hook(hook_pt, noise_hook)
# model.reset_hooks(including_permanent=True)
_, cache = model.run_with_cache(prompt, return_type="logits")

all_patterns = torch.stack([cache[f'blocks.{layer}.attn.hook_pattern'][0] for layer in range(12)])
all_patterns.shape

px.imshow(all_patterns.view(144, 15, 15)[:, -1, :].detach().cpu(), x=toks).show()

# %%
corrupted_prompt = "When Mary and John went to the shops. John gave the bag to"
corrupted_tokens = model.to_tokens(corrupted_prompt, prepend_bos=True)

model.add_perma_hook(hook_pt, noise_hook)
_, clean_cache = model.run_with_cache(prompt, return_type="logits")

def ioi_metric(logits, answer_token_indices=answer_idxs[:1]):
    return get_logit_diff(logits, answer_token_indices)

attn_head_out_all_pos_act_patch_results = patching.get_act_patch_attn_head_out_all_pos(model, corrupted_tokens, clean_cache, ioi_metric)
# %%
fig = px.imshow(attn_head_out_all_pos_act_patch_results.cpu().detach(),
    #    yaxis="Layer", 
    #    xaxis="Head",
       x=[f'{head}' for head in range(model.cfg.n_heads)],
       title="attn_head_out Activation Patching (All Pos)")
fig.show()

# %% 
print(b_noise.pow(2).mean())
px.imshow(b_noise.view(-1, 32).cpu().detach(), color_continuous_midpoint=0, color_continuous_scale='RdBu').show()