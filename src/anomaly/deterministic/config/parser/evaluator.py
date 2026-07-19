from ..config.evaluator import (
    LossCfg,
    EvaluatorCfg,
)


def loss(cfg):
    return LossCfg(
        name=cfg["loss"]["name"],
        params=cfg["loss"].get("params") or dict(),
    )


def evaluator(cfg):
    return EvaluatorCfg(
        loss=loss(cfg),
        percentile=cfg["evaluator"]["percentile"],
    )