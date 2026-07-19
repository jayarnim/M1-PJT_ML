import pandas as pd
import torch
from torchmetrics.classification import (
    BinaryAccuracy, 
    BinaryPrecision, 
    BinaryRecall,
    BinaryF1Score, 
    BinaryConfusionMatrix
)


class Calculator(object):
    def __init__(
        self, 
        scores: torch.Tensor, 
        percentile: list[float],
    ):
        super().__init__()
        self.scores = scores
        self.percentile = percentile
        self._set_up_components()

    def __call__(self, result):
        metrics = [
            self._metrics(result, p)
            for p in self.percentile
        ]
        return pd.DataFrame(metrics)

    def _metrics(self, result, p):
        kwargs = dict(
            input=self.scores, 
            q=p,
        )
        threshold = torch.quantile(**kwargs).cpu()

        y_pred = (result["score"] > threshold).int()
        y_true = result["label"]

        self.confmat.reset()
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()

        confmat = self.confmat(y_pred, y_true).cpu().numpy()

        return dict(
            p=p,
            threshold=threshold.item(),
            tp=confmat[1,1],
            tn=confmat[0,0],
            fp=confmat[0,1],
            fn=confmat[1,0],
            accuracy=self.accuracy(y_pred, y_true).item(),
            precision=self.precision(y_pred, y_true).item(),
            recall=self.recall(y_pred, y_true).item(),
            f1=self.f1(y_pred, y_true).item(),
        )

    def _set_up_components(self):
        self.confmat = BinaryConfusionMatrix()
        self.accuracy = BinaryAccuracy()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
        self.f1 = BinaryF1Score()