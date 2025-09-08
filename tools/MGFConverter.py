import os
import re
import pprint as pp
import numpy as np
import h5py, hdf5plugin

import xmltodict
import json

from pyteomics import mgf
from pyteomics.mgf import MGF

class MGFConverter(object):
    def __init__(self, meta):
        super().__init__()
        self.meta = meta
        self.pattern = r'([A-Z<>])([+-]\d+(?:\.\d+)?(?:[+-]\d+(?:\.\d+)?)*)'
        self.pattern2 = r'([A-Z<>][+-]\d+(?:\.\d+)?(?:[+-]\d+(?:\.\d+)?)*)'
        self.pattern3 = r'[+-]\d+(?:\.\d+)?'
        self._mass_to_ptm = None
        self._ptm_to_mass = None

    def index_mgf(self, mgf_file, xml_file=None):
        json_file = mgf_file.replace('.mgf', '-mgf-byte-offsets.json')
        if os.path.exists(json_file):
            print('Index file already exists, skipping the step of indexing...')
        else:
            lib = mgf.IndexedMGF(
                mgf_file,
                index_by_scans=True,
                read_schema=False  # 不解析额外字段，提速
            )
            lib.write_byte_offsets()

    def capture_mods_in_digits(self, seq=None):
        seq='<'+seq+'>'
        mods = re.findall(self.pattern, seq)
        result = [(name, [x for x in re.findall(r'[+-]\d+(?:\.\d+)?', nums)]) for name, nums in mods]
        return(result)

    def split_with_regex(self, seq):
        seq='<'+seq+'>'
        tokens = [t for t in re.split(self.pattern2, seq) if t.strip()]
        tokenized_seq=[]
        for t in tokens:
            m = re.match(self.pattern, t)
            if(m):
                name = m.group(1)
                nums = re.findall(self.pattern3, m.group(2))
                mods = [self._mass_to_ptm[n] for n in nums]
                mods.insert(0,name)
                token = '+'.join(mods)
                tokenized_seq.append(token)
            else:
                tokenized_seq += list(t)
        tokenized_seq = ','.join(tokenized_seq)
        return(tokenized_seq)

    def extract_ptms(self, mgf_file, ptm_file=None):
        ptms=dict()
        for spectrum in mgf.read(mgf_file):
            seq = spectrum['params']['seq']
            mods=self.capture_mods_in_digits(seq)
            for name, nums in mods:
                for ptm in nums:
                    if ptm not in ptms:
                        ptms[ptm]=1
                    ptms[ptm]+=+1

        if(ptm_file):
            f_out=open(ptm_file, 'w')
            for k in ptms.keys():
                f_out.write(k+'\n')
            f_out.close()

    def readin_mass_ptm_dicts(self, mass_ptm_file):
        mass_to_ptm=dict()
        ptm_to_mass=dict()
        f_in=open(mass_ptm_file, 'r')
        for line in f_in:
            line=line.strip()
            if(not line.startswith('#')):
                arr=line.split('\t')
                mass_to_ptm[arr[0]]=arr[1]
                ptm_to_mass[arr[1]]=arr[0]
        f_in.close()
        self._mass_to_ptm=mass_to_ptm
        self._ptm_to_mass=ptm_to_mass

        return(mass_to_ptm, ptm_to_mass)

    def convert_MassiveMGF_to_spec(self, mgf_file, spec_file=None, dryrun=False):
        if(not spec_file):
            spec_file=mgf_file+'.spec'
        if(dryrun):
            return(spec_file)


        f_out=open(spec_file, 'w')
        f_out.write('#Scan\tPeptide\tMass\tCharge\tRTinseconds\tIons(mz:intensity)\n')

        with MGF(mgf_file, convert_arrays=False, dtype=object) as reader:
            for spectrum in reader:
                scan = spectrum['params']['scan']
                charge = int(spectrum['params']['charge'][0])

                pepmass = spectrum['params']['pepmass'][0]
                precursor_mass = pepmass * charge - self.meta.proton * charge

                seq = spectrum['params']['seq']
                tokenized_seq = self.split_with_regex(seq)

                mz_array = spectrum['m/z array']                 # numpy.ndarray
                it_array = spectrum['intensity array']           # numpy.ndarray

                if(len(mz_array)!=len(it_array)):
                    sys.exit('Length of mz array and intensity array mismatch')

                peaks=[]
                for mz, it in zip(mz_array, it_array):
                    peaks.append(str(mz)+':'+str(it))
                peaks=','.join(peaks)
                output_line = [scan, tokenized_seq, str(precursor_mass), str(charge), '-', peaks]
                f_out.write('\t'.join(output_line) + '\n')
        f_out.close()

        return(spec_file)

    def initial_h5(self, hdf_file, mode):
        h5 = None
        if(mode=='w'):
            h5 = h5py.File(hdf_file,'w')
        elif(mode=='r'):
            h5 = h5py.File(hdf_file,'r')
        elif(mode=='a'):
            h5 = h5py.File(hdf_file,'a')
        else:
            raise ValueError('Invalid mode')
        return(h5)

    def convert_spec_to_h5(self, spec_file: str, dataset: str='default', chunk_size=1000, dryrun=False):
        hdf_file=spec_file+'.h5'
        if(dryrun):
            return(hdf_file)

        h5 = self.initial_h5(hdf_file, mode='w')
        dset = h5.require_dataset(name=dataset, shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
        buffer = []
        f_in=None
        if (spec_file.endswith('.gz')):
            f_in = gzip.open(spec_file, 'rt')
        else:
            f_in = open(spec_file, 'r')

        for line in f_in:
            spectrum = line.strip()
            m = re.search('^#', spectrum)
            if (m or spectrum == ''):
                continue
            buffer.append(spectrum)

            if len(buffer) >= chunk_size:
                spectra = np.array(buffer)
                new_shape = (dset.shape[0] + spectra.shape[0],)
                dset.resize(new_shape)
                dset[-spectra.shape[0]:] = spectra
                buffer = []  # Clear the buffer

        if (buffer):
            spectra = np.array(buffer)
            new_shape = (dset.shape[0] + spectra.shape[0],)
            dset.resize(new_shape)
            dset[-spectra.shape[0]:] = spectra

        f_in.close()

        return(hdf_file)

    def convert_mgf_to_spec(self, input_file, output_file=None):
        if(output_file is None):
            bn=os.path.basename(input_file)
            output_file = bn+'.spec'

        f_out=open(output_file, 'w')
        f_out.write('#Title\tPeptide\tMass\tCharge\tRTinseconds\tIons(mz:intensity)\n')

        total_peptide_num=0
        with mgf.MGF(input_file) as reader:
            for spectrum in reader:
                params = spectrum.get('params', {})
                title = params['title']
                rtinseconds = params.get('rtinseconds','-')
                rtinseconds = str(rtinseconds)

                pepmass = params['pepmass']
                pepmass = pepmass[0]
                charge = params['charge']
                charge = int(charge[0])

                precursor_mass = pepmass * charge - self.meta.proton * charge

                mz = spectrum['m/z array']
                mz = mz.astype(str)
                intensity = spectrum['intensity array']
                intensity = intensity.astype(str)
                peaks = list(zip(mz, intensity))
                peaks = [':'.join(i) for i in peaks]
                ions = ','.join(peaks)

                out_line = [title, '-', precursor_mass, charge, rtinseconds, ions]
                out_line = '\t'.join([str(i) for i in out_line])
                f_out.write(out_line+'\n')
                total_peptide_num+=1
        return(total_peptide_num)