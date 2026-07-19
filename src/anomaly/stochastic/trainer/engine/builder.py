from .loss.builder import loss_fn_builder
from .annealer.builder import annealer_builder
from .optimizer.builder import optimizer_builder
from .engine import Engine


def engine_builder(model, cfg):
    kwargs = dict(
        model=model,
        cfg=cfg.optimizer,
    )
    optimizer = optimizer_builder(**kwargs)

    kwargs = dict(
        cfg=cfg.loss,
    )
    criterion = loss_fn_builder(**kwargs)

    kwargs = dict(
        cfg=cfg.annealer,
    )
    annealer = annealer_builder(**kwargs)

    kwargs = dict(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        annealer=annealer,
    )
    return Engine(**kwargs)
