from dataclasses import dataclass
from typing import Literal
from .pipeline import PipelineCfg
from .trainer import TrainerCfg
from .evaluator import EvaluatorCfg
from .model import ModelCfg


@dataclass
class Config:
    model: ModelCfg
    pipeline: PipelineCfg
    trainer: TrainerCfg
    evaluator: EvaluatorCfg
    model_cls: Literal["vae", "dvae"]
    seed: int