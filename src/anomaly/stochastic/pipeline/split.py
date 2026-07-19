import pandas as pd
from sklearn.model_selection import train_test_split


def split_builder(
    df: pd.DataFrame, 
    y_col: str,
    cfg,
):
    normal = df[df[y_col]==0]
    
    tst_abnormal = df[df["Class"]==1]

    kwargs = dict(
        n=len(tst_abnormal),
        random_state=cfg.seed,
    )
    tst_normal = normal.sample(**kwargs)

    kwargs = dict(
        objs=[tst_normal, tst_abnormal],
        ignore_index=True,
    )
    tst = pd.concat(**kwargs)

    opt_normal = normal.drop(tst_normal.index)

    kwargs = dict(
        train_size=cfg.ratio["trn"],
        test_size=cfg.ratio["val"],  
        shuffle=cfg.shuffle,
        random_state=cfg.seed,
    )
    trn, val = train_test_split(opt_normal, **kwargs)

    return dict(
        trn=trn, 
        val=val, 
        tst=tst,
    )