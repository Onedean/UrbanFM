import torch
from torch.utils.data import Dataset


class ExpertForecastingDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class FoundationTrainDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


class ForecastingTestDataset(Dataset):
    def __init__(self, x, mask):
        self.x = x
        self.mask = mask

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.mask[idx]


class ImputationTestDataset(Dataset):
    def __init__(self, x, eval, mask):
        self.x = x
        self.eval = eval
        self.mask = mask

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.eval[idx], self.mask[idx]


