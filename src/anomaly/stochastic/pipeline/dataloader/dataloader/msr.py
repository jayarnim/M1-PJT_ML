import torch
from torch.utils.data import DataLoader, Dataset


class MSRDataset(Dataset):
    def __init__(self, X, y):
        kwargs = dict(
            data=X.values, 
            dtype=torch.float32,
        )
        self.X = torch.tensor(**kwargs)

        kwargs = dict(
            data=y.values, 
            dtype=torch.float32,
        )
        self.y = torch.tensor(**kwargs)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def msr_dataLoader(df, y_col, batch_size, shuffle):
        X_col = df.columns.difference([y_col])
        
        kwargs = dict(
            X=df.loc[:, X_col],
            y=df.loc[:, [y_col]],
        )
        dataset = MSRDataset(**kwargs)

        kwargs = dict(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,           
        )
        return DataLoader(**kwargs)