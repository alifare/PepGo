import h5py, hdf5plugin
import torch
from torch.utils.data import DataLoader
import pprint as pp

class HDF(torch.utils.data.Dataset):
    def __init__(self, hdf_file, dataset: str = 'default', workers=None):
        super().__init__()
        self.hdf_file = hdf_file
        self.dataset = dataset
        self.workers = workers

        self.h5 = h5py.File(self.hdf_file, 'r', swmr=True)
        self.dset=self.h5[dataset]

        print(self.__len__())
        '''
        seq=self.dset[0]
        print(seq)
        print('-'*100)
        pp.pprint(self.__getitem__(0))
        '''

    def __getitem__(self, idx):
        line = self.dset[idx]
        sample = self._parse_line(line)
        return (sample)

    def __len__(self):
        size = self.dset.size
        return (size)

    def _parse_line(self, line):
        line = line.decode().strip()
        print('line:')
        print(line)
        arr=line.split('\t')

        #scan=arr[0]
        peptide=arr[1]
        mass=float(arr[2])
        charge=int(arr[3])
        ions=arr[-1].split(',')
        x = [ [float(j) for j in i.split(':')] for i in ions ]
        y = peptide.split(',')

        sample = [x, y, [mass], [charge]]
        return (sample)