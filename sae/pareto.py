# %%
import wandb
import pandas as pd
import plotly.express as px

# %%
api = wandb.Api()

# anthropic = api.run("/tdooms/sae/runs/cw2fbm53")

anthropic = api.run("/tdooms/sae/runs/gik9on6k")
vanilla = api.run("/tdooms/sae/runs/bw2dj9li")
gated = api.run("/tdooms/sae/runs/zl8f0x9b")
rainbow = api.run("/tdooms/sae/runs/zef0t0e7")

runs = [rainbow, gated, vanilla, anthropic]
# %%
dfs = [run.history(samples=16_000) for run in runs]

# %%
# get training dynamics
o_df = pd.concat(dfs, keys=["rainbow", "gated", "vanilla", "anthropic"])
o_df["type"] = o_df.index.get_level_values(0)

df = o_df[~o_df['patch_loss/0'].isna()]
stacked = pd.DataFrame(dict(patch=[], l0=[], idx=[]))

sparsities = (0.01, 0.02, 0.04, 0.07, 0.14, 0.27, 0.52, 1.00)

col_map = lambda i: {f'patch_loss/{i}': 'patch', f"l0/{i}": 'l0', 'type': 'type'}

df = pd.concat([df[col_map(i).keys()].rename(columns=col_map(i)) for i in range(len(sparsities))], keys=sparsities)
df['sparsity'] = df.index.get_level_values(0)

# %%
px.line(df, x="l0", y="patch", color="sparsity", symbol="type", markers=True, title="Training Dynamics", log_x=True, log_y=True).update_layout(title_x=0.5)
# %%

df_last_epoch = df.groupby(['type', 'sparsity']).last().reset_index()
# df_last_epoch
px.line(df_last_epoch, x="l0", y="patch", color="type",markers=True, title="Pareto Frontiers", log_x=True, log_y=True).update_layout(title_x=0.5)

# %%