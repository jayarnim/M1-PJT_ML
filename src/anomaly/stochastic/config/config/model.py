from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelCfg:
    name: Literal["vae", "dvae"]
    params: dict