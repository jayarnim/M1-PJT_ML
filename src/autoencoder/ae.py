import torch.nn as nn
from .components.encoder import Encoder
from .components.latent import DeterministicSpace
from .components.decoder import Decoder
from .components.reconn import ReconstructionLayer


class AutoEncoder(nn.Module):
    def __init__(
        self, 
        input_dim, 
        latent_dim,
        output_dim,
        hidden_dim=None, 
        dropout=None, 
    ):
        super().__init__()
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self._set_up_components()

    def forward(self, X):
        # ENCODED
        h_encoder = self.encoder(X)
        # LATENT SPACE
        Z = self.latent(h_encoder)
        # DECODED
        h_decoder = self.decoder(Z)
        # RECONN
        X_hat = self.reconn(h_decoder)
        return X_hat

    def _set_up_components(self):
        self._create_modules()

    def _create_modules(self):
        # ===== ENCODER =====
        kwargs = dict(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        self.encoder = Encoder(**kwargs)

        # ===== LATENT SPACE =====
        FEAT_DIM = (
            self.hidden_dim[-1]
            if self.hidden_dim is not None
            else self.input_dim
        )
        kwargs = dict(
            feat_dim=FEAT_DIM,
            latent_dim=self.latent_dim,
        )
        self.latent = DeterministicSpace(**kwargs)

        # ===== DECODER =====
        kwargs = dict(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim[::-1],
            dropout=self.dropout,
        )
        self.decoder = Decoder(**kwargs)

        # ===== RECONN =====
        FEAT_DIM = (
            self.hidden_dim[0]
            if self.hidden_dim is not None
            else self.latent_dim
        )
        kwargs = dict(
            feat_dim=FEAT_DIM,
            output_dim=self.output_dim,
        )
        self.reconn = ReconstructionLayer(**kwargs)

