import pandas as pd
from .split import split_builder
from .dataloader.builder import dataloader_builder


def pipeline_builder(
    df: pd.DataFrame,
    y_col: str,
    cfg,
):
    kwargs = dict(
        df=df,
        y_col=y_col,
        cfg=cfg.split,
    )
    split = split_builder(**kwargs)

    dataloader = dict()

    for split_name, split_vals in split.items():
        TASK = (
            "msr"
            if split_name=="tst"
            else "opt"
        )

        kwargs = dict(
            df=split_vals,
            y_col=y_col,
            task=TASK,
            cfg=cfg.dataloader,
        )
        dataloader[split_name] = dataloader_builder(**kwargs)

    return dataloader