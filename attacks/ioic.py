# %%
%load_ext autoreload
%autoreload 2

from transformer_lens import HookedTransformer, HookedSAETransformer, HookedSAEConfig, HookedSAE
from transformer_lens import utils
from transformer_lens import patching
from transformer_lens.utils import download_file_from_hf
import torch
import plotly.express as px
import itertools
from einops import *

sae_names = [
    "gpt2-small_L0_Hcat_z_lr1.20e-03_l11.80e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L1_Hcat_z_lr1.20e-03_l18.00e-01_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v5",
    "gpt2-small_L2_Hcat_z_lr1.20e-03_l11.00e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v4",
    "gpt2-small_L3_Hcat_z_lr1.20e-03_l19.00e-01_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L4_Hcat_z_lr1.20e-03_l11.10e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v7",
    "gpt2-small_L5_Hcat_z_lr1.20e-03_l11.00e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L6_Hcat_z_lr1.20e-03_l11.10e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L7_Hcat_z_lr1.20e-03_l11.10e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L8_Hcat_z_lr1.20e-03_l11.30e+00_ds24576_bs4096_dc1.00e-05_rsanthropic_rie25000_nr4_v6",
    "gpt2-small_L9_Hcat_z_lr1.20e-03_l11.20e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L10_Hcat_z_lr1.20e-03_l11.30e+00_ds24576_bs4096_dc1.00e-05_rsanthropic_rie25000_nr4_v9",
    "gpt2-small_L11_Hcat_z_lr1.20e-03_l13.00e+00_ds24576_bs4096_dc3.16e-06_rsanthropic_rie25000_nr4_v9",
]
# %%
model = HookedSAETransformer.from_pretrained("gpt2-small")
# %%

corrupt_format = [
    "When{} and{} went to the shops.{} gave the bag banana",
    "When{} and{} went to the park.{} gave the ball banana",
    "When{} and{} went to the shops.{} gave an apple banana",
    "When{} and{} went to the park.{} gave a drink banana",
]
clean_format = [
    "When{} and{} went to the shops.{} gave the bag to",
    "When{} and{} went to the park.{} gave the ball to",
    "When{} and{} went to the shops.{} gave an apple to",
    "When{} and{} went to the park.{} gave a drink to",
]

# %%

utils.test_prompt("When John and Mary went to the park. John gave a drink banana to", " Mary", model)
# print(len(model.to_str_tokens(prompt_format[3].format(" John", " John"))))

# %%
names = [
    (" John", " Mary", ),
    (" Tom", " James", ),
    (" Dan", " Mark", ),
    (" Martin", " Amy", ),
    # (" Joseph", " Will"),
    # (" Bob", " Jack")
]


clean_prompts, corrupt_prompts = [], []
answer_tokens = []

for prompt, name in itertools.product(clean_format, names):
    clean_prompts += [prompt.format(name[0], name[1], name[0]), prompt.format(name[1], name[0], name[1])] 
    answer_tokens += [(name[1], name[0]), (name[0], name[1])]

for prompt, name in itertools.product(corrupt_format, names):
    corrupt_prompts += [prompt.format(name[0], name[1], name[1], name[0]), prompt.format(name[1], name[0], name[0], name[1])]
    # answer_tokens += [(name[1], name[0]), (name[0], name[1])]

# prompts = clean_prompts + corrupt_prompts
# %%
# %%
# prompts = [model.to_str_tokens(prompt) for prompt in corrupt_prompts]
# print([len(p) for p in prompts])

def get_logit_diff(logits, answer_token_indices):
    answer_token_indices = answer_token_indices.to(logits.device)
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits)

# answers: list of tuples (correct continuation, wrong continuation)
# tokens: input tokens
def get_logit_diffs(prompts):
    tokens = model.to_tokens(prompts, prepend_bos=True)
    # Get token indices answer tokens
    answer_token_indices = torch.tensor([[model.to_single_token(answer_tokens[i][j]) for j in range(2)] for i in range(len(answer_tokens))], device=model.cfg.device)
    # Clean logits
    logits = model(tokens)
    # Compute logit diff
    return get_logit_diff(logits, answer_token_indices)

print(f"Clean logit diff: {get_logit_diffs(clean_prompts).mean():.4f}")
print(f"Corrupt logit diff: {get_logit_diffs(corrupt_prompts).mean():.4f}")

px.line(torch.stack([get_logit_diffs(clean_prompts), get_logit_diffs(corrupt_prompts)]).cpu().detach()[..., 0].mT, title="Logit difference per prompt").show()

# %%
# List of prompts
# prompts = []
# # List of answers, in the format (correct, incorrect)
# answers = []
# # List of the token (ie an integer) corresponding to each answer, in the format (correct_token, incorrect_token)
# answer_tokens = []
# for i in range(len(prompt_format)):
#     for j in range(2):
#         answers.append((names[i][j], names[i][1 - j]))
#         answer_tokens.append(
#             (
#                 model.to_single_token(answers[-1][0]),
#                 model.to_single_token(answers[-1][1]),
#             )
#         )
#         prompts.append(prompt_format[i].format("", answers[-1][1]))
#         print(answer_tokens)
        
# answer_tokens = torch.tensor(answer_tokens)
# print(prompts)
# print(answers)

# positive = "John and Mary went to the park. Joseph had a good day. John gave a drink to."
# negative = "John and Mary went to the park. Mary had a good day. John gave a drink to."

# print(len(model.to_str_tokens(positive)), len(model.to_str_tokens(negative)))
# utils.test_prompt(positive, "Mary", model)

attn_layer = 9
resid_layer = 7
resid_hook_pt = utils.get_act_name("resid_pre", resid_layer)
attn_hook_pt = utils.get_act_name("z", attn_layer)

attn_sae_hook_pt = attn_hook_pt + ".hook_sae_acts_pre"
resid_sae_hook_pt = resid_hook_pt + ".hook_sae_acts_pre"

# %%
resid_hf_repo = "jbloom/GPT2-Small-SAEs-Reformatted" 
attn_hf_repo = "ckkissane/attn-saes-gpt2-small-all-layers"

resid_cfg_hf = download_file_from_hf(resid_hf_repo, f"{resid_hook_pt}/cfg.json")
attn_cfg_hf = download_file_from_hf(attn_hf_repo, f"{sae_names[attn_layer]}_cfg.json")

# %%

def res_sae_cfg_to_hooked_sae_cfg(sae_cfg):
    new_cfg = {
        "d_sae": sae_cfg["d_sae"],
        "d_in": sae_cfg["d_in"],
        "hook_name": sae_cfg["hook_point"],
    }
    return HookedSAEConfig.from_dict(new_cfg)
    
def attn_sae_cfg_to_hooked_sae_cfg(attn_sae_cfg):
    new_cfg = {
        "d_sae": attn_sae_cfg["dict_size"],
        "d_in": attn_sae_cfg["act_size"],
        "hook_name": attn_sae_cfg["act_name"],
    }
    return HookedSAEConfig.from_dict(new_cfg)

# %%
resid_cfg = res_sae_cfg_to_hooked_sae_cfg(resid_cfg_hf)
resid_state_dict = download_file_from_hf(resid_hf_repo, file_name=f"{resid_hook_pt}/sae_weights.safetensors")

resid_sae = HookedSAE(resid_cfg)
resid_sae.load_state_dict(resid_state_dict)
# %%
attn_cfg = attn_sae_cfg_to_hooked_sae_cfg(attn_cfg_hf)
attn_state_dict = download_file_from_hf(attn_hf_repo, file_name=f"{sae_names[attn_layer]}.pt", force_is_torch=True)

attn_sae = HookedSAE(attn_cfg)
attn_sae.load_state_dict(attn_state_dict)
# %%

# clean_loss = model.run_with_cache(positive, names_filter=attn_hook_pt, return_type="loss", loss_per_token=True)

# %%

_, clean_sae_cache = model.run_with_cache_with_saes(
    clean_prompts,
    saes = [resid_sae],
    names_filter=resid_sae_hook_pt,
    return_type="loss",
    loss_per_token=True
)

_, corrupt_sae_cache = model.run_with_cache_with_saes(
    corrupt_prompts,
    saes = [resid_sae],
    names_filter=resid_sae_hook_pt,
    return_type="loss",
    loss_per_token=True
)

print(clean_sae_cache[resid_sae_hook_pt].shape, corrupt_sae_cache[resid_sae_hook_pt].shape)

# px.line(torch.stack([clean_loss[0][0], sae_loss[0]]).cpu().detach().T, title="Loss per token")

# %%

# attn_sae
# (sae_cache[sae_hook_pt] > 0).sum(-1)
clean_sae_acts = clean_sae_cache[resid_sae_hook_pt][:, -5].relu()
corrupt_sae_acts = corrupt_sae_cache[resid_sae_hook_pt][:, -5].relu()

sae_acts = torch.cat([clean_sae_acts, corrupt_sae_acts], dim=0).cpu().detach()

live_feature_mask = sae_acts.relu() > 0
live_feature_union = live_feature_mask.any(dim=0)

half = rearrange(sae_acts[:, live_feature_union], "(i h) f -> i h f", i=2).mean(1)
values, indices = (half[1] - half[0]).cpu().detach().sort(-1)

px.line(
    y=values,
    x=list(map(str, live_feature_union.nonzero().flatten()[indices].tolist())),
    title=resid_hook_pt
).show()

px.imshow(
    sae_acts[..., live_feature_union].view(64, -1),
    title = f"Activations of Live SAE features at {resid_hook_pt}",
    x=list(map(str, live_feature_union.nonzero().flatten().tolist())),
    color_continuous_midpoint=0, color_continuous_scale='RdBu'
)
# %%

pos = resid_sae.W_dec[13526]

vals = einsum(pos, resid_sae.W_dec, "i, j i -> j")
values, indices = vals.sort()

px.line(y=values.detach().cpu())
# torch.nn.functional.cosine_similarity(pos, neg, dim=0)

# %%

toks = model.to_tokens(corrupt_prompts, prepend_bos=True)
model.to_str_tokens(toks[:, -5])
# %%
