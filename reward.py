import torch

def pilco_reward(x, c=None, Q=None):
    # x: [batch, state_dim]
    state_dim = x.shape[1]
    if c is None:
        c = torch.zeros(state_dim, dtype=x.dtype, device=x.device)
    if Q is None:
        Q = torch.eye(state_dim, dtype=x.dtype, device=x.device)
    diff = x - c
    cost = torch.exp(-0.5 * torch.sum((diff @ Q) * diff, dim=-1))
    return cost
