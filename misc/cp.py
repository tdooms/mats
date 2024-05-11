from torch import nn
import torch
from transformer_lens import utils
from dataclasses import dataclass
from einops import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import wandb
from tqdm import tqdm
from collections import namedtuple
from torch.optim import Adam

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from einops import rearrange
from transformer_lens import utils

def get_splits(n_vals=32):
    training = load_dataset("NeelNanda/c4-tokenized-2b", split="train", streaming=True).with_format("torch")

    validation = list(training.take(n_vals))
    validation = torch.stack([row["tokens"] for row in validation])
    
    return training, validation
                
class Sampler:
    def __init__(self, config, dataset, model):
        self.config = config
        self.model = model
        
        self.d_model = model.cfg.d_model
        self.n_ctx = model.cfg.n_ctx

        assert config.buffer_size % (config.in_batch * self.n_ctx) == 0, "samples must be a multiple of loader batch size"
        self.n_inputs = config.buffer_size // (config.in_batch * self.n_ctx)

        self.loader = DataLoader(dataset, batch_size=config.in_batch)
        self.batches = []

    def collect(self):
        result = rearrange(torch.stack(self.batches, dim=0), "... d_model -> (...) d_model")
        self.batches = []
        return result

    def extract(self, batch):
        hook_pt = utils.get_act_name(self.config.point, self.config.layer)
        _, cache = self.model.run_with_cache(batch, names_filter=[hook_pt], return_type="loss")
        return cache[hook_pt]

    @torch.inference_mode()
    def __iter__(self):
        self.batches = []

        for batch in self.loader:
            self.batches.append(self.extract(batch["tokens"]))
            
            if len(self.batches) == self.n_inputs:
                yield self.collect()
                
Loss = namedtuple('Loss', ['reconstruction', 'sparsity', 'auxiliary'])

@dataclass
class Config:
    buffer_size: int = 2**18  # ~250k tokens
    n_buffers: int = 100      # ~25M tokens
    
    # transformer_lens specific
    point : str = "resid_mid"
    layer : int = 0

    in_batch: int = 32
    out_batch: int = 4096

    expansion: int = 4
    lr: float = 1e-4
    
    validation_interval: int = 1000
    not_active_thresh: int = 2

    sparsities: tuple = (0.01, 0.1, 1)
    device = "cuda"
    
    magnitude: float = 0.3
    sparsity: float = 0.95

class BaseSAE(nn.Module):
    """
    Base class for all Sparse Auto Encoders.
    Provides a common interface for training and evaluation.
    """
    def __init__(self, config, model) -> None:
        super().__init__()
        self.config = config
        
        self.d_model = model.cfg.d_model
        self.n_ctx = model.cfg.n_ctx
        self.d_hidden = self.config.expansion * self.d_model
        self.n_instances = len(config.sparsities)
        
        self.steps_not_active = torch.zeros(self.n_instances, self.d_hidden)
        self.sparsities = torch.tensor(config.sparsities).to(config.device)
        self.step = 0
    
    def decode(self, x):
        return x
    
    def encode(self, x):
        return x
    
    def forward(self, x):
        x_hid, *_ = self.encode(x)
        return self.decode(x_hid)
    
    def loss(self, x, x_hid, x_hat, steps, *args):
        pass
    
    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        state = torch.load(path)
        new = cls(*args, **kwargs)
        new.load_state_dict(state)
        return new
    
    def calculate_metrics(self, x_hid, losses, *args):
        activeness = x_hid.sum(0)
        self.steps_not_active[activeness > 0] = 0
        
        metrics = dict(step=self.step)
        
        for i in range(self.n_instances):
            metrics[f"dead_fraction/{i}"] = (self.steps_not_active[i] > 2).float().mean().item()
            
            metrics[f"reconstruction_loss/{i}"] = losses.reconstruction[i].item()
            metrics[f"sparsity_loss/{i}"] = losses.sparsity[i].item()
            metrics[f"auxiliary_loss/{i}"] = losses.auxiliary[i].item()
            
            metrics[f"l1/{i}"] = x_hid[..., i, :].sum(-1).mean().item()
            metrics[f"l0/{i}"] = (x_hid[..., i, :] > 0).float().sum(-1).mean().item()
        
        self.steps_not_active += 1
        
        return metrics
    
    def preprocess(self, x):
        return x
    
    def train(self, sampler, model, validation, log=True):
        if log: wandb.init(project="sae")
        
        self.step = 0
        steps = self.config.n_buffers * (self.config.buffer_size // self.config.out_batch)

        scheduler = LambdaLR(self.optimizer, lr_lambda=lambda t: min(5*(1 - t/steps), 1.0))

        for buffer, _ in tqdm(zip(sampler, range(self.config.n_buffers)), total=self.config.n_buffers):
            loader = DataLoader(buffer, batch_size=self.config.out_batch, shuffle=True, drop_last=True)
            for x in loader:
                
                x = self.preprocess(x)
                x = repeat(x, "... d -> ... inst d", inst=self.n_instances)
                
                x_hid, *rest = self.encode(x)
                x_hat = self.decode(x_hid)
                
                losses = self.loss(x, x_hid, x_hat, self.step / steps, *rest)
                metrics = self.calculate_metrics(x_hid, losses, self.step, *rest)

                loss = (losses.reconstruction + self.sparsities * losses.sparsity + losses.auxiliary).sum()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                if self.step % self.config.validation_interval == 0:
                    clean_loss, losses = self.patch_loss(model, validation)
                    metrics |= {f"patch_loss/{i}": (loss.item() - clean_loss) / clean_loss for i, loss in enumerate(losses)}

                if log: wandb.log(metrics)
                self.step += 1
        
        if log: wandb.finish()
                
    @torch.inference_mode()
    def patch_loss(self, model, validation):
        losses = torch.zeros(self.n_instances, device=self.config.device)
        hook_pt = utils.get_act_name(self.config.point, self.config.layer)

        baseline, cache = model.run_with_cache(validation, return_type="loss", names_filter=[hook_pt])
        
        x = repeat(cache[hook_pt], "... d -> ... inst d", inst=self.n_instances)
        x_hat = self.forward(x)

        # run model with recons patched in per instance
        for inst_id in range(self.n_instances):
            patch_hook = lambda act, hook: x_hat[:, :, inst_id]
            loss = model.run_with_hooks(validation, return_type="loss", fwd_hooks = [(hook_pt, patch_hook)])
            losses[inst_id] = loss.item()

        return baseline, losses

class AnthropicSAE(BaseSAE):
    """
    The main difference between this is the loss function.
    Specifically, it uses the activation * the output norm as the sparsity term.
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    """
    def __init__(self, config, model):
        super().__init__(config, model)
        device = config.device

        W_dec = torch.randn(self.n_instances, self.d_hidden, self.d_model, device=device)
        W_dec /= torch.norm(W_dec, dim=-1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)

        W_enc = W_dec.mT.clone().to(device)
        self.W_enc = nn.Parameter(W_enc)

        # Contrary to Anthropic, we actually still use a decoder norm because it seems more logical.
        self.b_enc = nn.Parameter(torch.zeros(self.n_instances, self.d_hidden, device=device))
        self.b_dec = nn.Parameter(torch.zeros(self.n_instances, self.d_model, device=device))

        directions = torch.randn(128, self.d_model, device=device) * config.magnitude
        mask = torch.rand_like(directions, device=device) > config.sparsity
        
        self.directions = nn.Parameter(mask.float() * directions, requires_grad=False)
        
        self.optimizer = Adam(self.parameters(), lr=self.config.lr, betas=(0.9, 0.999))

    def preprocess(self, x):
        noise = self.directions[torch.arange(x.size(-2), device=x.device) % 128]
        return x + noise
        
    def encode(self, x):
        return torch.relu(einsum(x - self.b_dec, self.W_enc, "... inst d, inst d hidden -> ... inst hidden") + self.b_enc),

    def decode(self, h):
        return einsum(h, self.W_dec, "... inst hidden, inst hidden d -> ... inst d") + self.b_dec

    def loss(self, x, x_hid, x_hat, fraction):
        reconstruction = ((x_hat - x) ** 2).mean(0).sum(dim=-1)

        norm = self.W_dec.norm(dim=-1)
        lambda_ = min(1, fraction * 20)
        
        sparsity = einsum(x_hid, norm, "batch inst hidden, inst hidden -> batch inst").mean(dim=0)

        return Loss(reconstruction, lambda_ * sparsity, torch.zeros(self.n_instances, device=x.device))