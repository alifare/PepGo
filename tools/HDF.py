import h5py, hdf5plugin
import torch
from torch.utils.data import DataLoader

class HDF(torch.utils.data.Dataset):
    def __init__(self, hdf_file=None, dataset: str = 'default', workers=None):
        super().__init__()
        self.hdf_file = hdf_file
        self.dataset = dataset
        self.workers = workers
        
        self.h5 = h5py.File(self.hdf_file, 'r', swmr=True)
        self.dset=self.h5[dataset]

    def __getitem__(self, idx):
        seq = self.dset[idx]
        return (seq)

    def __len__(self):
        size = self.dset.size
        return (size)