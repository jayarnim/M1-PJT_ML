import torch.nn as nn


class ReconstructionLayer(nn.Module):
    def __init__(
        self, 
        feat_dim, 
        output_dim,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.output_dim = output_dim
        self._set_up_components()

    def forward(self, h):
        return self.linear(h)

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            in_features=self.feat_dim,
            out_features=self.output_dim,
        )
        self.linear = nn.Linear(**kwargs)