import torch
import torch.nn as nn


class Noiser(nn.Module):
    def __init__(
        self, 
        noise: float,
    ):
        super().__init__()
        self.dropout = nn.Dropout(noise)

    def forward(
        self, 
        X: torch.Tensor,
    ) -> torch.Tensor:
        return self.dropout(X)