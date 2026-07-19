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
        
        self.counter = 0
        self.should_stop = False

        self._current_epoch = 0
        self._current_score = None
        self._current_state = None
        self._current_anomaly = None

        self._best_epoch = 0
        self._best_score = float("inf")
        self._best_state = None
        self._best_anomaly = None

    def __call__(
        self,
        current_score: float,
        current_state: collections.OrderedDict,
        current_anomaly: torch.Tensor,
    ):
        self._current_epoch += 1
        self._current_state = current_state
        self._current_anomaly = current_anomaly

        IMPROVED = current_score < self.best_score - self.delta
        
        if self._current_epoch <= self.warmup:
            self.should_stop = False
            self.counter = 0
        
        else:
            if IMPROVED:
                self._best_score = current_score
                self._best_state = current_state
                self._best_anomaly = current_anomaly
                self._best_epoch = self.current_epoch
                self.counter = 0
            else:
                self.counter += 1

        if self.counter > self.patience:
            self.should_stop = True

    @property
    def best_epoch(self):
        return (
            self._best_epoch
            if self._best_epoch!=0
            else self._current_epoch
        )

    @property
    def best_score(self):
        return (
            self._best_score
            if self._best_score is not None
            else self._current_score
        )

    @property
    def best_state(self):
        return (
            self._best_state
            if self._best_state is not None
            else self._current_state
        )

    @property
    def best_anomaly(self):
        return (
            self._best_anomaly
            if self._best_anomaly is not None
            else self._current_anomaly
        )