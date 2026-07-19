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

    normal = dict()
    anomaly = dict()

    for base, models in zip(BASE_MODELS, MODELS):
        mapper = dict(zip(models, DROPOUTS))
        
        data0 = {
            mapper[model]: dfs[model][dfs[model]["label"]==0]["score"].values
            for model in models
        }
        normal[base] = pd.DataFrame(data=data0)

        data1 = {
            mapper[model]: dfs[model][dfs[model]["label"]==1]["score"].values
            for model in models
        }
        anomaly[base] = pd.DataFrame(data=data1)

    return normal, anomaly