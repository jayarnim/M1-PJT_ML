from IPython.display import clear_output
import copy
import torch


# device setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(
        self, 
        engine,
        monitor,
        num_epochs,
    ):
        super().__init__()
        self.engine = engine
        self.monitor = monitor
        self.num_epochs = num_epochs

    def fit(
        self, 
        trn_loader: torch.utils.data.dataloader.DataLoader, 
        val_loader: torch.utils.data.dataloader.DataLoader, 
    ):
        kwargs = dict(
            trn_loader=trn_loader, 
            val_loader=val_loader, 
        )
        records = self.progressor(**kwargs)

        model_state = (
            self.monitor.best_state
            if self.monitor.best_state is not None
            else self.monitor.current_state
        )
        self.model.load_state_dict(model_state)

        clear_output(wait=False)

        print(
            "VALIDATION",
            f"\tBEST SCORE: {self.monitor.best_score:.4f}",
            f"\tBEST EPOCH: {self.monitor.best_epoch}",
            sep="\n",
        )

        return records

    def progressor(self, trn_loader, val_loader):
        trn_log_list = dict(
            nll=list(),
            kl=list(),
        )
        val_log_list = dict(
            nll=list(),
            kl=list(),
        )

        for epoch in range(self.num_epochs):
            # RUN ==========
            kwargs = dict(
                trn_loader=trn_loader, 
                val_loader=val_loader,
            )
            trn_result, val_result = self.engine(**kwargs)

            kwargs = dict(
                current_score=val_result["loss"],
                current_state=copy.deepcopy(self.model.state_dict()),
                current_anomaly=val_result["anomaly"],
            )
            self.monitor(**kwargs)

            # ACCUMULATE ==========
            trn_log_list["nll"].append(trn_result["nll"])
            trn_log_list["kl"].append(trn_result["kl"])
            val_log_list["nll"].append(val_result["nll"])
            val_log_list["kl"].append(val_result["kl"])

            # EARLY STOPPING ==========
            if self.monitor.should_stop==True:
                break

            # PRINT ==========
            print(
                f'CURRENT TRN NLL: {trn_result["nll"]:.4f}',
                f'KL: {trn_result["kl"]:.4f}',
                sep='\t\t',
            )
            print(
                f'CURRENT VAL NLL: {val_result["nll"]:.4f}',
                f'KL: {val_result["kl"]:.4f}',
                f'COUNTER: {self.monitor.counter}',
                sep='\t\t',
            )

            # LOG RESET ==========
            if (epoch + 1) % 50 == 0:
                clear_output(wait=False)

        anomaly = (
            self.monitor.best_anomaly
            if self.monitor.best_anomaly is not None
            else val_result["anomaly"]
        )

        return dict(
            trn=trn_log_list,
            val=val_log_list,
            anomaly=anomaly,
        )

    @property
    def model(self):
        return self.engine.model