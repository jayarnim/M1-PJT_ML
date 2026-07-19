from typing import Literal
import pandas as pd
from .dataloader.registry import DATALOADER_REGISTRY


def dataloader_builder(
    df: pd.DataFrame,
    y_col: str,
    task: Literal["opt", "msr"],
    cfg,
):
    kwargs = dict(
        df=df,
        y_col=y_col,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
    )
    func = DATALOADER_REGISTRY[task]
    return func(**kwargs)