from ..config.trainer import (
    OptimizerCfg,
    LossCfg,
    EngineCfg,
    MonitorCfg,
    TrainerCfg,
)


def optimizer(cfg):
    return OptimizerCfg(
        name=cfg["optimizer"]["name"],
        params=cfg["optimizer"].get("params") or dict(),
    )


def loss(cfg):
    return LossCfg(
        name=cfg["loss"]["name"],
        params=cfg["loss"].get("params") or dict(),
    )


def engine(cfg):
    return EngineCfg(
        optimizer=optimizer(cfg),
        loss=loss(cfg),
    )


def monitor(cfg):
    return MonitorCfg(
        delta=cfg["monitor"]["delta"],
        patience=cfg["monitor"]["patience"],
        warmup=cfg["monitor"]["warmup"],
    )


def trainer(cfg):
    return TrainerCfg(
        num_epochs=cfg["trainer"]["num_epochs"],
        engine=engine(cfg),
        monitor=monitor(cfg),
    )