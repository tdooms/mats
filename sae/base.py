from torch import nn
import torch
from transformer_lens import utils
from dataclasses import dataclass
from einops import *
from torch.utils.data import DataLoader

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import wandb
from tqdm import tqdm
from collections import namedtuple
from jaxtyping import Float
from torch import Tensor

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

    whiten: bool = False
    strip_first_n: int = 0
    eps : float = 0.01

    n_buffers_for_cov : int = 64

class BaseSAE(nn.Module):
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
        
        self.cov = None
    
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
    def from_pretrained(cls, path, device='cpu', *args):
        state = torch.load(path)
        new = cls(*args)
        return new.load_state_dict(state).to(device)
    
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

    def preprocess_acts(self, x):
        if self.config.whiten:
            return einsum(x-self.mean, self.inv_sqrt_cov, "... d_model1, d_model1 d_model2 -> ... d_model2")
        else:
            return x

    def postprocess_acts(self, x_hat):
        if self.config.whiten:
            return einsum(x_hat, self.sqrt_cov, "... d_model1, d_model1 d_model2 -> ... d_model2") +self.mean
        else:
            return x_hat
        
    def compute_covariance_and_mean(self, sampler):
        running_mean = torch.zeros([self.d_model], device=self.config.device)
        running_cov = torch.zeros([self.d_model, self.d_model], device=self.config.device)
        for buffer, _ in tqdm(zip(sampler.sample(), range(self.config.n_buffers_for_cov)), total=self.config.n_buffers_for_cov):
            mean = buffer.mean(dim=0)
            centered = buffer - mean
            cov = einsum(centered, centered, "batch d_model1, batch d_model2 -> d_model1 d_model2") / buffer.size(0)
            
            running_mean += mean
            running_cov += cov
        
        self.mean = running_mean / self.config.n_buffers_for_cov
        self.cov = running_cov / (self.config.n_buffers_for_cov - 1)
        u, s, v = torch.linalg.svd(self.cov)
        self.sqrt_cov = u @ torch.diag((s + self.config.eps).sqrt()) @ v.T
        self.inv_sqrt_cov = u @ torch.diag((s + self.config.eps).rsqrt()) @ v.T
    
    def train(self, sampler, model, validation, log=True):
        if log: wandb.init(project="sae")
        self.compute_covariance_and_mean(sampler)
        self.step = 0
        steps = self.config.n_buffers * (self.config.buffer_size // self.config.out_batch)

        scheduler = LambdaLR(self.optimizer, lr_lambda=lambda t: min(5*(1 - t/steps), 1.0))

        for buffer, _ in tqdm(zip(sampler.sample(), range(self.config.n_buffers)), total=self.config.n_buffers):

            loader = DataLoader(buffer, batch_size=self.config.out_batch, shuffle=True, drop_last=True)
            for x in loader:
                
                x = self.preprocess_acts(x)
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

                if log and (self.step % self.config.validation_interval == 0):
                    clean_loss, losses = self.patch_loss(model, validation)
                    metrics |= {f"patch_loss/{i}": (losses[i].item() - clean_loss) / clean_loss for i, loss in enumerate(losses)}

                if log: wandb.log(metrics)
                self.step += 1
        
        if log: wandb.finish()
                
    #@torch.inference_mode()
    def patch_loss(self, model, validation):
        losses = torch.zeros(self.n_instances, device=self.config.device)
        hook_pt = utils.get_act_name(self.config.point, self.config.layer)

        baseline, cache = model.run_with_cache(validation, return_type="loss", names_filter=[hook_pt], loss_per_token=True)
        baseline = baseline[:, self.config.strip_first_n:].mean().item()

        x = self.preprocess_acts(cache[hook_pt])
        x = repeat(x, "... d -> ... inst d", inst=self.n_instances)
        x_hat = self.forward(x)
        x_hat = self.postprocess_acts(x_hat)

        # run model with recons patched in per instance
        for inst_id in range(self.n_instances):
            def patch_hook(act, hook):
                new_act = act.clone()
                new_act[:,self.config.strip_first_n:] = x_hat[:, self.config.strip_first_n:, inst_id].clone()
                return new_act
            loss = model.run_with_hooks(validation, return_type="loss", fwd_hooks = [(hook_pt, patch_hook)], loss_per_token=True)
            losses[inst_id] = loss[:, self.config.strip_first_n:].mean().item()

    
        return baseline, losses
    

    