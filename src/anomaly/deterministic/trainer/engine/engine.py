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
    ):
        super().__init__()
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.criterion = criterion
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
        trn_log = self.epoch_step(**kwargs)

        kwargs = dict(
            dataloader=val_loader,
            task="VAL",
        )
        with torch.no_grad():
            self.model.eval()
            val_log = self.epoch_step(**kwargs)

        return trn_log, val_log

    def epoch_step(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader,
        task: Literal["TRN", "VAL"],
    ):
        epoch_anomaly = []
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
                batch_anomaly = self.batch_step(X)
                batch_loss = batch_anomaly.mean()

            # backward pass
            if task=="TRN":
                self.backprop(batch_loss)

            # accumulate loss
            epoch_anomaly.append(batch_anomaly.detach().cpu())
            epoch_loss += batch_loss.item()

        return dict(
            anomaly=torch.cat(tensors=epoch_anomaly, dim=0),
            loss=epoch_loss / len(dataloader),
        )

    def batch_step(self, X):
        X_hat = self.model(X)
        score = self.criterion(X_hat, X)
        return score

    def backprop(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()