import torch.nn as nn


class Noiser(nn.Module):
    def __init__(
        self, 
        noise,
    ):
        super().__init__()
        self.dropout = nn.Dropout(noise)

    def forward(self, X):
        return self.dropout(X)