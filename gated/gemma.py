# %%

from transformer_lens import HookedTransformer

# %%

model = HookedTransformer.from_pretrained("qwen1.5-0.5b")

# %%

model("Write me a poem about Machine Learning.")

# %%