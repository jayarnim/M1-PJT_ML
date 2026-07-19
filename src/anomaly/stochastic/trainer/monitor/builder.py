from .monitor import Monitor


def monitor_builder(cfg):
    kwargs = dict(
        delta=cfg.delta,
        patience=cfg.patience,
        warmup=cfg.warmup,
    )
    return Monitor(**kwargs)