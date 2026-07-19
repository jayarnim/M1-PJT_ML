import torch
from .layers.noiser import Noiser
from .layers.encoder import Encoder
from .layers.latent import StochasticSpace
from .layers.decoder import Decoder
from .layers.reconn import ReconstructionLayer
from components.base import BayesModel
from components.base import BayesModelOutput


class DenoisedVariationalAutoEncoder(BayesModel):
    def __init__(
        self, 
        input_dim: int, 
        latent_dim: int,
        output_dim: int,
        noise: float,
        hidden_dim: list[int]=None, 
        dropout: float=None, 
    ):
        super().__init__(locals())

        # ===== NOISER =====
        kwargs = dict(
            noise=noise,
        )
        self.noiser = Noiser(**kwargs)

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
        # NOISE INJECTION
        X_tilde = self.noiser(X)
        # ENCODED
        h_encoder = self.encoder(X_tilde)
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
