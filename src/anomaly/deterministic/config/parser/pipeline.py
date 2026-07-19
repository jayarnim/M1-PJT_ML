from ..config.pipeline import (
    SplitCfg,
    DataloaderCfg,
    PipelineCfg,
)


def split(cfg):
    return SplitCfg(
        ratio=cfg["split"]["ratio"],
        shuffle=cfg["split"]["shuffle"],
        seed=cfg["seed"],
    )


def dataloader(cfg):
    return DataloaderCfg(
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=cfg["dataloader"]["shuffle"],
    )


def pipeline(cfg):
    return PipelineCfg(
        split=split(cfg),
        dataloader=dataloader(cfg),
    )