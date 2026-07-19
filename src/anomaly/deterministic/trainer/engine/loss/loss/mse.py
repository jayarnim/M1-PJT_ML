import torch.nn.functional as F


class MeanSquaredError(object):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, X_hat, X):
        kwargs = dict(
            input=X_hat, 
            target=X, 
            reduction='none',
        )
        return F.mse_loss(**kwargs).mean(dim=1)