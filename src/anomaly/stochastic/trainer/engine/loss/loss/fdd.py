import torch.nn.functional as F


class FusedDirectionalDistance(object):
    def __init__(self, lamb):
        super().__init__()
        self.lamb = lamb

    def __call__(self, X_hat, X):
        kwargs = dict(
            input=X_hat, 
            target=X, 
            reduction='none',
        )
        mse = F.mse_loss(**kwargs).mean(dim=1)

        kwargs = dict(
            x1=X_hat, 
            x2=X, 
            dim=1,
        )
        dir = F.cosine_similarity(**kwargs)

        return self.lamb * mse + (1-self.lamb) * (1-dir)