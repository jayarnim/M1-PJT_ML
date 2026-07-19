from tqdm import tqdm
import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor(object):
    def __init__(
        self, 
        model: nn.Module,
        criterion,
    ):
        super().__init__()
        self.model = model.to(DEVICE)
        self.criterion = criterion

    @torch.no_grad()
    def __call__(
        self, 
        tst_loader: torch.utils.data.dataloader.DataLoader,
    ):
        self.model.eval()

        label_list = []
        score_list = []

        kwargs = dict(
            iterable=tst_loader, 
            desc=f"TST",
        )
        for X, label in tqdm(**kwargs):
            X = X.to(DEVICE)
            X_hat, kl = self.model(X)
            score = self.criterion(X_hat, X)
            score_list.extend(score.cpu().tolist())
            label_list.extend(label.cpu().tolist())

        return dict(
            score=torch.tensor(data=score_list, dtype=torch.float32).squeeze(-1),
            label=torch.tensor(data=label_list, dtype=torch.int64).squeeze(-1),
        )