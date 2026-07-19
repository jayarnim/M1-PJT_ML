from typing import Literal
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Engine(object):
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        criterion,
        annealer,
    ):
        super().__init__()
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.criterion = criterion
        self.annealer = annealer
        self.scaler = GradScaler(device=DEVICE)
        self.current_epoch = 0

    def __call__(
        self, 
        trn_loader: torch.utils.data.dataloader.DataLoader, 
        val_loader: torch.utils.data.dataloader.DataLoader, 
    ):
        self.current_epoch += 1

        kwargs = dict(
            dataloader=trn_loader,
            task="TRN",
        )
        self.model.train()
        trn_result = self.epoch_step(**kwargs)

        kwargs = dict(
            dataloader=val_loader,
            task="VAL",
        )
        with torch.no_grad():
            self.model.eval()
            val_result = self.epoch_step(**kwargs)

        return trn_result, val_result

    def epoch_step(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader,
        task: Literal["TRN", "VAL"],
    ):
        epoch_anomaly = []
        epoch_nll = 0.0
        epoch_kl = 0.0
        epoch_loss = 0.0

        kwargs = dict(
            iterable=dataloader, 
            desc=f"EPOCH {self.current_epoch} {task}"
        )
        for X in tqdm(**kwargs):
            # to gpu
            X = X.to(DEVICE)

            # forward pass
            with autocast(DEVICE.type):
                batch_nll, batch_kl = self.batch_step(X)
                batch_elbo = batch_nll + self.annealer(self.current_epoch) * batch_kl
                
                batch_anomaly = batch_nll.detach().cpu()
                batch_nll = batch_nll.mean()
                batch_kl = batch_kl.mean()
                batch_loss = batch_elbo.mean()

            # backward pass
            if task=="TRN":
                self.backprop(batch_loss)

            # accumulate loss
            epoch_anomaly.append(batch_anomaly)
            epoch_nll += batch_nll.item()
            epoch_kl += batch_kl.item()
            epoch_loss += batch_loss.item()

        return dict(
            anomaly=torch.cat(tensors=epoch_anomaly, dim=0),
            nll=epoch_nll / len(dataloader), 
            kl=epoch_kl / len(dataloader),
            loss=epoch_loss / len(dataloader),
        )

    def batch_step(self, X):
        X_hat, kl = self.model(X)
        nll = self.criterion(X_hat, X)
        return nll, kl

    def backprop(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()