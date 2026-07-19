import torch
import torch.nn as nn
from components.functions import fc_block


class Decoder(nn.Module):
    def __init__(
        self, 
        latent_dim: int,
        hidden_dim: list[int]=None, 
        dropout: float=None,
    ):
        super().__init__()
        # ===== MULTI-LAYER =====
        if hidden_dim is not None:
            kwargs = dict(
                input_dim=latent_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            components = list(fc_block(**kwargs))
            self.mlp = nn.Sequential(*components)
        
        # ===== SINGLE LAYER =====
        else:
            self.mlp = nn.Identity()

    def forward(
        self, 
        Z: torch.Tensor,
    ) -> torch.Tensor:
        return self.mlp(Z)