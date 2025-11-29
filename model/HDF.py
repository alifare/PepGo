import h5py, hdf5plugin
import torch
from torch.utils.data import DataLoader
import pprint as pp

class HDF(torch.utils.data.Dataset):
    def __init__(self, hdf_file, dataset: str = 'default', reverse=False):
        super().__init__()
        self.hdf_file = hdf_file
        self.dataset = dataset
        self.reverse = reverse

        with h5py.File(self.hdf_file, 'r') as h5:
            self.dset_shape = h5[dataset].shape
            self.dset_size = h5[dataset].size

    def __len__(self):
        return(self.dset_size)

    def __getitem__(self, idx):
        # 每次读取时打开文件（支持多进程）
        with h5py.File(self.hdf_file, 'r') as h5:
            dset = h5[self.dataset]
            line = dset[idx]
            sample = self._parse_line(line)
            return(sample)

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
        return(sample)
