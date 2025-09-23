import h5py, hdf5plugin
import torch
from torch.utils.data import DataLoader
import pprint as pp

class HDF(torch.utils.data.Dataset):
    def __init__(self, hdf_file, dataset: str = 'default', workers=None, reverse=False):
        super().__init__()
        self.hdf_file = hdf_file
        self.dataset = dataset
        self.workers = workers
        self.reverse = reverse

        self.h5 = h5py.File(self.hdf_file, 'r', swmr=True)
        self.dset = self.h5[dataset]

    def __getitem__(self, idx):
        line = self.dset[idx]
        sample = self._parse_line(line)
        return (sample)

    def __len__(self):
        size = self.dset.size
        return (size)

    def _parse_line(self, line):
        line = line.decode().strip()
        arr = line.split('\t')

        peptide = arr[1]
        mass = float(arr[2])
        charge = int(arr[3])
        ions = arr[-1].split(',')
        x = [ [float(j) for j in i.split(':')] for i in ions ]
        y = peptide.split(',')
        if(self.reverse):
            y = y[::-1]

        sample = [x, y, [mass], [charge]]
        return (sample)