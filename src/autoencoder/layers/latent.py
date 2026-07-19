import torch
import torch.nn as nn
import torch.distributions as dist


class DeterministicSpace(nn.Module):
    def __init__(
        self, 
        feature_dim: int, 
        latent_dim: int,
    ):
        super().__init__()
        kwargs = dict(
            in_features=feature_dim,
            out_features=latent_dim,
        )
        self.latent = nn.Linear(**kwargs)

    def forward(
        self, 
        h: torch.Tensor,
    ) -> torch.Tensor:
        return self.latent(h)


class StochasticSpace(nn.Module):
    def __init__(
        self, 
        feature_dim: int, 
        latent_dim: int,
    ):
        super().__init__()
        # ===== VARIATIONAL DIST. PARAM LAYER =====
        kwargs = dict(
            in_features=feature_dim,
            out_features=latent_dim,
        )
        self.mu = nn.Linear(**kwargs)
        self.logvar = nn.Linear(**kwargs)

        # ===== PRIOR DIST. =====
        self.p = dist.Normal(loc=0, scale=1)

    def forward(
        self, 
        h: torch.Tensor,
    ) -> torch.Tensor:
        # ===== PARAMS =====
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)

        # ===== VARIATIONAL DIST. =====
        q = dist.Normal(loc=mu, scale=std)

        # ===== KL DIVERGENCE =====
        kld = dist.kl_divergence(p=q, q=self.p).mean(dim=1)

        # ===== SAMPLING =====
        Z = (
            q.rsample()
            if self.training
            else mu
        )
        
        return Z, kld