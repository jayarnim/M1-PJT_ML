import yaml
from pathlib import Path
from .parser.parser import parser


def config_builder(path):
    path = Path(path)

    kwargs = dict(
        file=path,
        mode="r",
        encoding="utf-8",
    )
    with open(**kwargs) as f:
        cfg = yaml.safe_load(f)

    return parser(cfg)