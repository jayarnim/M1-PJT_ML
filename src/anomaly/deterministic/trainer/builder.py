from .engine.builder import engine_builder
from .monitor.builder import monitor_builder
from .trainer import Trainer


def trainer_builder(model, cfg):
    kwargs = dict(
        model=model,
        cfg=cfg.engine,
    )
    engine = engine_builder(**kwargs)

    kwargs = dict(
        cfg=cfg.monitor,
    )
    monitor = monitor_builder(**kwargs)

    kwargs = dict(
        engine=engine,
        monitor=monitor,
        num_epochs=cfg.num_epochs,
    )
    return Trainer(**kwargs)