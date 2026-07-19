from dataclasses import dataclass
from typing import Literal


@dataclass
class OptimizerCfg:
    name: Literal["adagrad", "adam", "adamw"]
    params: dict


@dataclass
class LossCfg:
    name: Literal["fdd"]
    params: dict


@dataclass
class AnnealerCfg:
    name: Literal["linear"]
    params: dict


@dataclass
class EngineCfg:
    optimizer: OptimizerCfg
    loss: LossCfg
    annealer: AnnealerCfg


@dataclass
class MonitorCfg:
    delta: float
    patience: int
    warmup: int


@dataclass
class TrainerCfg:
    num_epochs: int
    engine: EngineCfg
    monitor: MonitorCfg