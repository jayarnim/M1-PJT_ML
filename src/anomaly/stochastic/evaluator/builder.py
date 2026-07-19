import torch
from .loss.builder import loss_fn_builder
from .predictor import Predictor
from .calculator import Calculator
from .evaluator import Evaluator


def evaluator_builder(model, scores, cfg):
    kwargs = dict(
        cfg=cfg.loss,
    )
    criterion = loss_fn_builder(**kwargs)

    kwargs = dict(
        model=model,
        criterion=criterion,
    )
    predictor = Predictor(**kwargs)

    kwargs = dict(
        scores=scores,
        percentile=cfg.percentile,
    )
    calculator = Calculator(**kwargs)

    kwargs = dict(
        predictor=predictor,
        calculator=calculator,
    )
    return Evaluator(**kwargs)