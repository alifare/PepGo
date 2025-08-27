import os
import re
import pprint as pp

import xmltodict
import json

from pyteomics import mgf

class MGFConverter(object):
    def __init__(self, meta):
        super().__init__()
        self.meta = meta
        self.pattern = r'([A-Z<>])([+-]\d+(?:\.\d+)?(?:[+-]\d+(?:\.\d+)?)*)'
        self.pattern2 = r'([A-Z<>][+-]\d+(?:\.\d+)?(?:[+-]\d+(?:\.\d+)?)*)'
        self.pattern3 = r'[+-]\d+(?:\.\d+)?'

    def capture_mods_in_digits(self, seq=None):
        seq='<'+seq+'>'
        mods = re.findall(self.pattern, seq)
        result = [(name, [x for x in re.findall(r'[+-]\d+(?:\.\d+)?', nums)]) for name, nums in mods]
        return(result)

    def split_with_regex(self, ptms, seq):
        seq='<'+seq+'>'
        tokens = [t for t in re.split(self.pattern2, seq) if t.strip()]
        tokenized_seq=[]
        for t in tokens:
            m = re.match(self.pattern, t)
            if(m):
                name = m.group(1)
                nums = re.findall(self.pattern3, m.group(2))
                mods = [ptms[n] for n in nums]
                mods.insert(0,name)
                token = '+'.join(mods)
                tokenized_seq.append(token)
            else:
                tokenized_seq += list(t)
        tokenized_seq = ','.join(tokenized_seq)
        return(tokenized_seq)

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
            '''
            # 随机访问 scan=12345
            spec = lib[12345]
            print(spec['params'])
            '''

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

    def convert_MassiveMGF_to_spec(self, mgf_file, mass_ptm_file=None):
        ptms=dict()
        if(mass_ptm_file):
            f_in=open(mass_ptm_file, 'r')
            for line in f_in:
                line=line.strip()
                if(not line.startswith('#')):
                    arr=line.split('\t')
                    ptms[arr[0]]=arr[1]
            f_in.close()
        #pp.pprint(ptms)

        for spectrum in mgf.read(mgf_file):
            params = spectrum['params']        # 标题
            seq = spectrum['params']['seq']
            charge = spectrum['params']['charge']

            mz = spectrum['m/z array']                 # numpy.ndarray
            it = spectrum['intensity array']           # numpy.ndarray
            tokenized_seq = self.split_with_regex(ptms, seq)
            print(seq)
            print(tokenized_seq)
            print('=' * 100)

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