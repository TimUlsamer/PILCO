import torch
import torch.nn as nn

class RBFPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, num_features=10):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_features, input_dim))
        self.log_widths = nn.Parameter(torch.zeros(num_features, input_dim))
        self.linear = nn.Linear(num_features, output_dim)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)  # [batch, 1, input_dim]
        centers = self.centers.unsqueeze(0)  # [1, num_features, input_dim]
        widths = torch.exp(self.log_widths).unsqueeze(0)
        phi = torch.exp(-0.5 * ((x - centers)**2 / (widths**2)).sum(-1))
        return self.linear(phi)
