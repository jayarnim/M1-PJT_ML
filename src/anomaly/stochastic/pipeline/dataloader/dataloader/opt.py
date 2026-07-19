import torch
from torch.utils.data import DataLoader, Dataset


class OPTDataset(Dataset):
    def __init__(self, X):
        kwargs = dict(
            data=X.values, 
            dtype=torch.float32,
        )
        self.X = torch.tensor(**kwargs)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def opt_dataLoader(df, y_col, batch_size, shuffle):
    X_col = df.columns.difference([y_col])
    dataset = OPTDataset(df.loc[:, X_col])

    kwargs = dict(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,           
    )
    return DataLoader(**kwargs)