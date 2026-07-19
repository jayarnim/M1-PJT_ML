import torch
from .layers.encoder import Encoder
from .layers.latent import StochasticSpace
from .layers.decoder import Decoder
from .layers.reconn import ReconstructionLayer
from components.base import BayesModel
from components.base import BayesModelOutput


class VariationalAutoEncoder(BayesModel):
    def __init__(
        self, 
        input_dim: int, 
        latent_dim: int,
        output_dim: int,
        hidden_dim: list[int]=None, 
        dropout: float=None, 
    ):
        super().__init__(locals())

        # ===== ENCODER =====
        kwargs = dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.encoder = Encoder(**kwargs)

        # ===== LATENT SPACE =====
        FEATURE_DIM = (
            hidden_dim[-1]
            if hidden_dim is not None
            else input_dim
        )
        kwargs = dict(
            feature_dim=FEATURE_DIM,
            latent_dim=latent_dim,
        )
        self.latent = StochasticSpace(**kwargs)

        # ===== DECODER =====
        kwargs = dict(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim[::-1],
            dropout=dropout,
        )
        self.decoder = Decoder(**kwargs)

        # ===== RECONN =====
        FEATURE_DIM = (
            hidden_dim[0]
            if hidden_dim is not None
            else latent_dim
        )
        kwargs = dict(
            feature_dim=FEATURE_DIM,
            output_dim=output_dim,
        )
        self.reconn = ReconstructionLayer(**kwargs)

    def forward(
        self, 
        X: torch.Tensor,
    ) -> torch.Tensor:
        # ENCODED
        h_encoder = self.encoder(X)
        # LATENT SPACE
        Z, kld = self.latent(h_encoder)
        # DECODED
        h_decoder = self.decoder(Z)
        # RECONN
        hat = self.reconn(h_decoder)
        return BayesModelOutput(
            hat=hat, 
            kld=kld,
        )
