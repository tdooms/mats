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
model = HookedTransformer.from_pretrained("pythia-160m", device="cuda")

dataset = load_dataset("NeelNanda/c4-10k", split="train").with_format("torch")
batch = model.tokenizer(dataset["text"][:16], return_tensors="pt", padding='max_length', truncation=True, max_length=256).to("cuda")["input_ids"]

# %%
clean_loss = model(batch, return_type="loss")
clean_loss
# %%

patch_hook = lambda act, hook: torch.zeros_like(act, device="cuda")

names = ["hook_resid_pre", "hook_resid_mid", "hook_resid_post", "mlp.hook_post"]

vals = {k: [] for k in names}

for i, n in itertools.product(range(model.cfg.n_layers), range(names)):
    hook_pt = f"blocks.{i}.{n}"

    loss = model.run_with_hooks(batch, return_type="loss", fwd_hooks = [(hook_pt, patch_hook)])
    vals[]
    print(f"corrupt_loss (layer {i}):", loss.item())
# %%

df = pd.DataFrame(dict(
    resid_mid = [16.1358, 19.3029, 15.6645, 14.3311, 17.5412, 15.4793, 10.216, 13.3566, 12.233, 9.5622, 9.979, 16.6478],
    resid_post = [7.9449, 8.3259, 12.4827, 18.4202, 15.7372, 13.2676, 14.379, 12.2468, 17.2821, 11.8794, 10.1842, 8.0666],
    mlp_post = [8.0517, 5.1053, 5.5695, 5.6138, 5.628, 5.5436, 5.5964, 5.5636, 5.6816, 5.7287, 5.7772, 5.2257],
    clean = [5.5648] * 12,
))

px.line(df, y=["clean", "resid_mid", "resid_post", "mlp_post"], x=df.index, title="Zero Ablation Loss (GPT-2)").update_layout(title_x=0.5, yaxis_title="Loss", xaxis_title="Layer")

