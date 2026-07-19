from dataclasses import dataclass
from typing import Literal


@dataclass
class LossCfg:
    name: Literal["fdd"]
    params: dict


@dataclass
class EvaluatorCfg:
    loss: LossCfg
    percentile: list[float]