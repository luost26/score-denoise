import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class ToyPointCloudDataset(Dataset):

    def __init__(self, shape='plane', num_pnts=10000, size=24, transform=None):
        super().__init__()
        assert shape in ('plane', 'sphere')
        self.shape = shape
        self.size = size
        self.num_pnts = num_pnts
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx >= self.size:
            raise IndexError()
        pcl = torch.rand([self.num_pnts, 3])
        if self.shape == 'plane':
            pcl[:, 2] = 0
        elif self.shape == 'sphere':
            pcl -= 0.5
            pcl /= (pcl ** 2).sum(dim=1, keepdim=True).sqrt()
        data = {
            'pcl_clean': pcl,
        }
        if self.transform is not None:
            data = self.transform(data)
        return data
