from typing import Literal
import torch
from torch import nn


# TopK activation function
# by Gao et al (2024)
# https://arxiv.org/abs/2406.04093
#
# Based on
# https://github.com/openai/sparse_autoencoder
# MIT license

class TopK(nn.Module):
    def __init__(self, 
            k: int, 
            activation: Literal["Identity", "ReLU", "Tanh", "Sigmoid"] = "Identity"
            ) -> None:
        super().__init__()
        self.k = k
        if activation == 'ReLU':
            self.postact_fn = nn.ReLU()
        elif activation == 'Tanh':
            self.postact_fn = nn.Tanh()
        elif activation == 'Sigmoid':
            self.postact_fn = nn.Sigmoid()
        else:
            self.postact_fn = nn.Identity()
    # Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result
