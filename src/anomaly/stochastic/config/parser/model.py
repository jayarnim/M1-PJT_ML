from ..config.model import (
    ModelCfg,
)


def model(cfg):
    return ModelCfg(
        name=cfg["model"]["name"],
        params=cfg["model"]["params"],
    )