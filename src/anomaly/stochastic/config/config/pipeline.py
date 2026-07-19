from dataclasses import dataclass


@dataclass
class SplitCfg:
    ratio: dict[str, int]
    shuffle: bool
    seed: int


@dataclass
class DataloaderCfg:
    batch_size: int
    shuffle: bool


@dataclass
class PipelineCfg:
    split: SplitCfg
    dataloader: DataloaderCfg