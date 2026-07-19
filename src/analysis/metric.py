from .const import (
    AE_MODELS,
    VAE_MODELS,
    BASE_MODELS,
    MODELS,
    DROPOUTS,
)
from pathlib import Path
import pandas as pd


def loader(path):
    path = Path(path)
    files = list(path.iterdir())
    
    dfs = {
        file.stem: pd.read_csv(file)
        for file in files
    }

    p = dfs["ae"]["p"]
    cols = dfs["ae"].columns.difference(["p"])

    result = dict()

    for col in cols:
        data = {
            name: df[col].values
            for name, df in dfs.items()
        }
        df = pd.DataFrame(data=data, index=p)

        result[col] = {
            base: df[model].rename(columns=dict(zip(model, DROPOUTS)))
            for base, model in zip(BASE_MODELS, MODELS)
        }

    return result