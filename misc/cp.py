# %%

from transformer_lens import HookedTransformer
from datasets import load_dataset
from transformer_lens import utils
import torch
import plotly.express as px
import pandas as pd
import itertools

# %%

torch.set_grad_enabled(False)
model = HookedTransformer.from_pretrained("gelu-4l", device="cuda")

dataset = load_dataset("NeelNanda/c4-10k", split="train").with_format("torch")
batch = model.tokenizer(dataset["text"][:16], return_tensors="pt", padding='max_length', truncation=True, max_length=256).to("cuda")["input_ids"]

# %%
clean_loss = model(batch, return_type="loss")
clean_loss
# %%

patch_hook = lambda act, hook: torch.zeros_like(act, device="cuda")

names = ["hook_resid_pre", "hook_resid_mid", "hook_resid_post", "mlp.hook_post"]

vals = {k: [] for k in names}

for i, n in itertools.product(range(model.cfg.n_layers), names):
    hook_pt = f"blocks.{i}.{n}"

    loss = model.run_with_hooks(batch, return_type="loss", fwd_hooks = [(hook_pt, patch_hook)])
    vals[n].append(loss.item())
# %%

vals

# px.line(df, y=["clean", "resid_pre", "resid_mid", "resid_post", "mlp_post"], x=df.index, title="Zero Ablation Loss (GPT-2)").update_layout(title_x=0.5, yaxis_title="Loss", xaxis_title="Layer")

