import torch
import torch.nn as nn
import torch.distributions as dist


class DeterministicSpace(nn.Module):
    def __init__(
        self, 
        feat_dim, 
        latent_dim,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self._set_up_components()

    def forward(self, h):
        return self.latent(h)

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            in_features=self.feat_dim,
            out_features=self.latent_dim,
        )
        self.latent = nn.Linear(**kwargs)


class StochasticSpace(nn.Module):
    def __init__(
        self, 
        feat_dim, 
        latent_dim,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self._set_up_components()

    def forward(self, h):
        # ===== PARAMS =====
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)

        # ===== VARIATIONAL DIST. =====
        q = dist.Normal(loc=mu, scale=std)

        # ===== KL DIVERGENCE =====
        kl = dist.kl_divergence(p=q, q=self.p).mean(dim=1)

        # ===== SAMPLING =====
        Z = (
            q.rsample()
            if self.training
            else mu
        )
        
        return Z, kl

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        # ===== VARIATIONAL DIST. PARAM LAYER =====
        kwargs = dict(
            in_features=self.feat_dim,
            out_features=self.latent_dim,
        )
        self.mu = nn.Linear(**kwargs)
        self.logvar = nn.Linear(**kwargs)

        # ===== PRIOR DIST. =====
        self.p = dist.Normal(loc=0, scale=1)