import torch.nn as nn
from ..functions.generator import fc_block


class Encoder(nn.Module):
    def __init__(
        self, 
        input_dim,
        hidden_dim=None, 
        dropout=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self._set_up_components()

    def forward(self, X):
        return self.mlp(X)

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        # ===== MULTI-LAYER =====
        if self.hidden_dim is not None:
            kwargs = dict(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
            )
            components = list(fc_block(**kwargs))
            self.mlp = nn.Sequential(*components)
        
        # ===== SINGLE LAYER =====
        else:
            self.mlp = nn.Identity()