import collections
import torch


class Monitor(object):
    def __init__(
        self,
        delta: float,
        patience: int,
        warmup: int,
    ):
        super().__init__()
        self.delta = delta
        self.patience = patience
        self.warmup = warmup
        
        self.current_epoch = 0
        self.best_epoch = 0
        self.best_score = float("inf")
        self.best_state = None
        self.best_anomaly = None
        self.current_state = None
        self.current_anomaly = None
        self.counter = 0
        self.should_stop = False

    def __call__(
        self,
        current_score: float,
        current_state: collections.OrderedDict,
        current_anomaly: torch.Tensor,
    ):
        self.current_epoch += 1
        self.current_state = current_state
        self.current_anomaly = current_anomaly

        IMPROVED = current_score < self.best_score - self.delta
        
        if (self.current_epoch) <= self.warmup:
            self.should_stop = False
            self.counter = 0
        
        else:
            if IMPROVED:
                self.best_score = current_score
                self.best_state = current_state
                self.best_anomaly = current_anomaly
                self.best_epoch = self.current_epoch
                self.counter = 0
            else:
                self.counter += 1

        if self.counter > self.patience:
            self.should_stop = True