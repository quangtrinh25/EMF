import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EMFDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.emf = torch.tensor(data['emf'], dtype=torch.float32)
        self.target = torch.tensor(data['target'], dtype=torch.float32)

    def __len__(self):
        return len(self.emf)

    def __getitem__(self, idx):
        return self.emf[idx], self.target[idx]

def get_dataloader(npz_path, batch_size, shuffle=True, num_workers=0):
    dataset = EMFDataset(npz_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
