# %%

import torch



x = torch.randn(3, requires_grad=True)
a = torch.randn(2, 3, requires_grad=True).detach()

target = torch.randn(2)

loss = (x @ a.T - target).pow(2).sum()
print(loss.item(), a.norm().item())

optim = torch.optim.Adam([x, a], lr=0.1)

optim.zero_grad()
loss.backward()
optim.step()

loss = (x @ a.T - target).pow(2).sum()
print(loss.item(), a.norm().item())