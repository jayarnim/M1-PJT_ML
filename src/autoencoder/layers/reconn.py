import torch
import torch.nn as nn


class ReconstructionLayer(nn.Module):
    def __init__(
        self, 
        feature_dim: int, 
        output_dim: int,
    ):
        super().__init__()
        kwargs = dict(
            in_features=feature_dim,
            out_features=output_dim,
        )
        self.linear = nn.Linear(**kwargs)

    def forward(
        self, 
        h: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(h)