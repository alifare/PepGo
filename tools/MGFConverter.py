import os
import sys
import re
import pprint as pp
import numpy as np
import h5py, hdf5plugin

import xmltodict
import json

import ast
import operator

from decimal import Decimal, ROUND_HALF_UP
import re

import re


# 支持的操作符
_OP_MAP = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

from pyteomics import mgf
from pyteomics.mgf import MGF

class MGFConverter(object):
    def __init__(self, meta, input_format=None, output_format=None):
        super().__init__()
        self._meta = meta
        self.pattern1 = r'([A-Z<>])([+-]\d+(?:\.\d+)?(?:[+-]\d+(?:\.\d+)?)*)'
        self.pattern2 = r'([A-Z<>][+-]\d+(?:\.\d+)?(?:[+-]\d+(?:\.\d+)?)*)'
        self.pattern3 = r'[+-]\d+(?:\.\d+)?'

        self.pattern4 = r'([A-Z<>])(\([+-]\d*(?:\.\d+)\)?(?:\([+-]\d*(?:\.\d+)\)?)*)'
        self.pattern5 = r'([A-Z<>]\([+-]\d*(?:\.\d+)\)?(?:\([+-]\d*(?:\.\d+)\)?)*)'
        self.pattern6 = r'\([+-]\d*(?:\.\d+)?\)'

        self._mass_to_ptm = None
        self._ptm_to_mass = None
        self.scan_table = None

        self._replace_isoleucine_and_leucine_with_X = self._meta.configs['Model']['Peptide']['replace_isoleucine_and_leucine_with_X']
        self._replace_isoleucine_with_leucine = self._meta.configs['Model']['Peptide']['replace_isoleucine_with_leucine']
        self.have_seen = dict()
        
        self._input_format = input_format
        self._output_format = output_format

        self._allowed_max_ptm_on_one_residue = self._meta.configs['Model']['Peptide']['allowed_max_ptm_on_one_residue']

        print('input_format',end=':\t')
        print(self._input_format)
        print('output_format',end=':\t')
        print(self._output_format)

        if((not self._input_format) or (not self._output_format)):
            raise ValueError('input_format or output_format must be specified!')

    def set_input_format(self, input_format=None):
        self._input_format = input_format
        
    def set_output_format(self, output_format=None):
        self._output_format = output_format

    def _eval(self, node):
        """递归求值 AST 节点（仅允许数字与四则运算）"""
        if isinstance(node, ast.Num):  # 3.14 、 7
            return node.n
        if isinstance(node, ast.BinOp):  # left + right
            return _OP_MAP[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp):  # -x
            return _OP_MAP[type(node.op)](self._eval(node.operand))
        raise ValueError("不支持的表达式")

    def safe_eval(self, expr: str) -> float:
        """安全计算字符串表达式"""
        node = ast.parse(expr, mode='eval').body
        return self._eval(node)

    def eval_with_precision_policy(self, expr: str) -> str:
        res = Decimal(str(self.safe_eval(expr)))  # 带符号的 Decimal
        p = max(-Decimal(str(self.safe_eval(m.group()))).as_tuple().exponent
                for m in re.finditer(r'[+-]?\d+(?:\.\d+)?', expr))
        q = Decimal('0.1') ** p
        # 用 'f' 格式化即可自动保留正负号
        out = format(res.quantize(q), '+f')
        # 去掉无意义的尾零和尾小数点，但保留负号
        #out = out.rstrip('0').rstrip('.')

        return(out)

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
        mods = re.findall(self.pattern1, seq)
        result = [(name, [x for x in re.findall(self.pattern3, nums)]) for name, nums in mods]

        return(result)

    def replace_mass_with_token(self, m):
        name = m.group(1)
        nums_str = m.group(2)

        if(self._input_format=='MassIVE_KB'):
            split_pattern=self.pattern2
            match_pattern=self.pattern1
            find_pattern=self.pattern3
        elif(self._input_format=='9species'):
            split_pattern=self.pattern5
            match_pattern=self.pattern4
            find_pattern=self.pattern6
        else:
            raise ValueError('The mode must be MassIVE_KB or 9species')

        nums = re.findall(find_pattern, nums_str)
        nums_size = len(nums)

        mods = [self._mass_to_ptm.get(n) for n in nums]
        if None in mods:
            return(None, None)

        '''
        print('self._mass_to_ptm',end=':')
        print(len(self._mass_to_ptm))
        pp.pprint(self._mass_to_ptm)
        print('name',end=':\t')
        print(name)
        print('mods',end=':\t')
        print(mods)
        print('nums_str',end=':\t')
        print(nums_str)
        sys.exit()
        '''

        if (self._replace_isoleucine_and_leucine_with_X and (name == 'I' or name == 'L')):
            name = 'X'

        if(self._output_format=='Casanovo'):
            if (nums_size > 1):
                nums = self.eval_with_precision_policy(nums_str)
            else:
                nums = mods[0]
            token = '[' + nums + ']'

            if (name == '<'):
                token = token + '-'
            else:
                token = name + token
        elif(self._output_format=='PointNovo'):
            if (nums_size > 1):
                nums = self.eval_with_precision_policy(nums_str)
            else:
                #nums = nums_str
                nums = mods[0]
            token = '(' + nums + ')'

            if (name == '<'):
                token = token
            else:
                token = name + token
        elif(self._output_format=='PrimeNovo'):
            token = nums_str
            if (name == '<'):
                token = token
            else:
                token = name + token
        elif (self._output_format == 'PepGo'):
            token = '+'.join(mods)
            if (name == '<'):
                token = token
            else:
                token = name +'+'+ token
        else:
            sys.exit('The output_format must be specified')

        if (False and nums_size > 1):
            if (nums_str not in self.have_seen):
                self.have_seen[nums_str] = 1
                print('name', end=':')
                print(name)
                print('nums_str', end=':')
                print(nums_str)
                print('nums', end=':')
                print(nums)
                print('mods', end=':')
                print(mods)
                print('token',end=':')
                print(token)
                print('-' * 100)

        return(token, nums_size)

    def modify_seq_to_format(self, seq):
        seq='<'+seq+'>'

        if(self._input_format=='MassIVE_KB'):
            split_pattern=self.pattern2
            match_pattern=self.pattern1
        elif(self._input_format=='9species'):
            split_pattern=self.pattern5
            match_pattern=self.pattern4
        else:
            raise ValueError('The input_format must be MassIVE_KB or 9species')

        char_join=''
        if(self._output_format == 'PepGo'):
            char_join = ','
        tokens = [t for t in re.split(split_pattern, seq) if t.strip()]

        max_ptm_on_one_residue=0

        tokenized_seq=[]
        for t in tokens:
            m = re.match(match_pattern, t)
            if(m):
                token, nums_size = self.replace_mass_with_token(m)
                if((token is None) or (nums_size is None)):
                    return(None, None)
                tokenized_seq.append(token)
                if(nums_size > max_ptm_on_one_residue):
                    max_ptm_on_one_residue = nums_size
            else:
                if(self._replace_isoleucine_and_leucine_with_X):
                    tokenized_seq += list(t.replace('I', 'X').replace('L', 'X'))
                else:
                    tokenized_seq += list(t)
        tokenized_seq[-1] = re.sub(r'[-+]>$', '', tokenized_seq[-1])
        if(tokenized_seq[0]=='<'):
            tokenized_seq = tokenized_seq[1:]
        if (tokenized_seq[-1] == '>'):
            tokenized_seq = tokenized_seq[:-1]

        tokenized_seq = char_join.join(tokenized_seq)

        return(tokenized_seq, max_ptm_on_one_residue)

    def extract_ptms(self, mgf_file, ptm_file=None):
        ptms=dict()
        for spectrum in mgf.read(mgf_file):
            seq = spectrum['params']['seq']
            mods=self.capture_mods_in_digits(seq)

            for name, nums in mods:
                name_and_ptm = name+'\t'+'\t'.join(nums)
                if(name_and_ptm not in ptms):
                    ptms[name_and_ptm]=1
                ptms[name_and_ptm]+=1

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

        print('self._mass_to_ptm')
        print(self._mass_to_ptm)

        return(mass_to_ptm, ptm_to_mass)

    def readin_mass_scan_table(self, scan_table_file):
        with open(scan_table_file, 'r', encoding='utf-8') as f:
            self.scan_table = {line.strip() for line in f}
        return(self.scan_table)

    def batch_write_to_MGF(self, input_mgf, output_mgf=None):
        spectra_buffer = []
        batch_size = 100  # 每100个谱图写入一次

        first_batch = True
        with MGF(input_mgf) as reader:
            mode = 'w'
            for spectrum in reader:
                #print('spectrum',end=':')
                #print(type(spectrum))
                #pp.pprint(spectrum)
                SCANS = spectrum['params']['scans']
                if((self.scan_table is not None) and (SCANS not in self.scan_table)):
                    continue

                seq = spectrum['params']['seq']
                tokenized_seq, max_ptm_on_one_residue = self.modify_seq_to_format(seq)

                if(self._output_format=='Casanovo'):
                    spectrum['params']['seq'] = tokenized_seq
                elif(self._output_format=='PrimeNovo'):
                    TITLE = spectrum['params']['provenance_filename']+','+ spectrum['params']['provenance_scan']
                    PEPMASS = spectrum['params']['pepmass']
                    CHARGE = spectrum['params']['charge']
                    RTINSECONDS = 0.0
                    SEQ = tokenized_seq
                    spectrum['params']={
                        'title':TITLE,
                        'pepmass':PEPMASS,
                        'charge':CHARGE,
                        'scans':SCANS,
                        'seq':SEQ,
                        'rtinseconds':RTINSECONDS
                    }

                spectra_buffer.append(spectrum)

                # 批量写入以减少内存使用
                if len(spectra_buffer) >= batch_size:
                    mode = 'w' if first_batch else 'a'
                    with open(output_mgf, mode) as f:
                        mgf.write(spectra_buffer, f)
                    spectra_buffer = []
                    first_batch = False

            # 写入剩余的谱图
            if spectra_buffer:
                with open(output_mgf, mode) as f:
                    mgf.write(spectra_buffer, f)


    def convert_MassiveMGF_to_PrimeNovo(self, mgf_file, PrimeNovo_mgf_file, dryrun=False):
        print('raw_mgf',end=':\t')
        print(mgf_file)
        print('PrimeNovo_mgf',end=':\t')
        print(PrimeNovo_mgf_file)
        if(not dryrun):
            self.batch_write_to_MGF(mgf_file, PrimeNovo_mgf_file)

        return(PrimeNovo_mgf_file)

    def convert_MassiveMGF_to_PointNovo(self, mgf_file, dryrun=False):
        PointNovo_mgf_file = mgf_file+'.PointNovo.mgf'
        PointNovo_csv_file = mgf_file+'.PointNovo.csv'

        print('mgf',end=':\t')
        print(mgf_file)
        print('PointNovo_mgf',end=':\t')
        print(PointNovo_mgf_file)
        print('PointNovo_csv',end=':\t')
        print(PointNovo_csv_file)

        if(not dryrun):
            f_out_s1 = open(PointNovo_mgf_file, 'w')
            f_out_f1 = open(PointNovo_csv_file, 'w')
            f_out_f1.write('spec_group_id,m/z,z,rt_mean,seq,scans,profile,feature area,irt\n')

            spectra_buffer = []
            batch_size = 100  # 每100个谱图写入一次

            first_batch = True
            with MGF(mgf_file) as reader:
                mode = 'w'
                for spectrum in reader:
                    #print('spectrum')
                    #pp.pprint(spectrum)
                    seq = spectrum['params']['seq']
                    seq = self.modify_seq_to_format(seq)
                    spectrum['params']['seq'] = seq

                    spec_group_id = spectrum['params']['scan']
                    mz = str(spectrum['params']['pepmass'][0])
                    z = str(spectrum['params']['charge'][0])
                    rt_mean = '0'

                    scans = spectrum['params']['scans']
                    feature_area = '10.0'
                    irt = '0'
                    profile = str(rt_mean) + ':' + str(feature_area)

                    #print(seq)
                    #print('-'*100)
                    csv_arr = [spec_group_id, mz, z, rt_mean, seq, scans, profile, feature_area, irt]
                    f_out_f1.write(','.join(csv_arr)+'\n')

                    spectra_buffer.append(spectrum)

                    # 批量写入以减少内存使用
                    if len(spectra_buffer) >= batch_size:
                        mode = 'w' if first_batch else 'a'
                        with open(PointNovo_mgf_file, mode) as f:
                            mgf.write(spectra_buffer, f)
                        spectra_buffer = []
                        first_batch = False

                # 写入剩余的谱图
                if spectra_buffer:
                    with open(PointNovo_mgf_file, mode) as f:
                        mgf.write(spectra_buffer, f)

            f_out_s1.close()
            f_out_f1.close()

        return(PointNovo_mgf_file, PointNovo_csv_file)

    def convert_MassiveMGF_to_CasanovoMGF(self, mgf_file, casanovomgf_file=None, dryrun=False):
        if(not casanovomgf_file):
            casanovomgf_file=mgf_file+'.casanovo.mgf'
        if(dryrun):
            return(casanovomgf_file)
        self.batch_write_to_MGF(mgf_file, output_mgf=casanovomgf_file)

        '''
        spectra_buffer = []
        batch_size = 100  # 每100个谱图写入一次

        have_seen=dict()
        first_batch = True
        with MGF(mgf_file) as reader:
            mode = 'w'
            for spectrum in reader:
                seq = spectrum['params']['seq']
                spectrum['params']['seq'] = self.modify_seq_to_format(seq)
                spectra_buffer.append(spectrum)

                # 批量写入以减少内存使用
                if len(spectra_buffer) >= batch_size:
                    mode = 'w' if first_batch else 'a'
                    with open(casanovomgf_file, mode) as f:
                        mgf.write(spectra_buffer, f)
                    spectra_buffer = []
                    first_batch = False

            # 写入剩余的谱图
            if spectra_buffer:
                with open(casanovomgf_file, mode) as f:
                    mgf.write(spectra_buffer, f)

        return(casanovomgf_file)
        '''

    def convert_9SpeciesMGF_to_PepGo(self, mgf_file, spec_file=None, dryrun=False, preprocess=False):
        if(not spec_file):
            spec_file=mgf_file+'.spec'
        if(dryrun):
            return(spec_file)

        f_out=open(spec_file, 'w')
        f_out.write('#Scans\tPeptide\tMass\tCharge\tRTinseconds\tIons(mz:intensity)\n')

        with MGF(mgf_file, convert_arrays=False, dtype=object) as reader:
            for spectrum in reader:
                if(preprocess):
                    spectrum = self._meta.preprocess_spectrum(spectrum)
                if(spectrum is None):
                    continue

                seq = spectrum['params']['seq']
                tokenized_seq, max_ptm_on_one_residue = self.modify_seq_to_format(seq)
                if(max_ptm_on_one_residue > self._allowed_max_ptm_on_one_residue):
                    continue

                mz_array = spectrum['m/z array']
                it_array = spectrum['intensity array']
                assert len(mz_array) == len(it_array), 'Length of mz array and intensity array mismatch!'

                scans = spectrum['params'].get('scans', None)
                charge = int(spectrum['params']['charge'][0])

                pepmass = spectrum['params']['pepmass'][0]
                precursor_mass = pepmass * charge - self._meta.proton * charge

                peaks=[]
                for mz, it in zip(mz_array, it_array):
                    peaks.append(str(mz)+':'+str(it))
                peaks=','.join(peaks)
                output_line = [str(scans), tokenized_seq, str(precursor_mass), str(charge), '-', peaks]
                f_out.write('\t'.join(output_line) + '\n')
        f_out.close()

        return(spec_file)

    def convert_MassiveKBmgf_to_PepGo(self, mgf_file, spec_file=None, dryrun=False, preprocess=False):
        if(not spec_file):
            spec_file=mgf_file+'.spec'
        if(dryrun):
            return(spec_file)

        f_out=open(spec_file, 'w')
        f_out.write('#Scan\tPeptide\tMass\tCharge\tRTinseconds\tIons(mz:intensity)\n')

        with MGF(mgf_file, convert_arrays=False, dtype=object) as reader:
            for spectrum in reader:
                if(preprocess):
                    spectrum = self._meta.preprocess_spectrum(spectrum)
                if(spectrum is None):
                    continue

                seq = spectrum['params']['seq']
                tokenized_seq, max_ptm_on_one_residue = self.modify_seq_to_format(seq)
                if((tokenized_seq is None) or (max_ptm_on_one_residue is None)):
                    continue
                if(max_ptm_on_one_residue > self._allowed_max_ptm_on_one_residue):
                    continue

                mz_array = spectrum['m/z array']                 # numpy.ndarray
                it_array = spectrum['intensity array']           # numpy.ndarray
                assert len(mz_array) == len(it_array), 'Length of mz array and intensity array mismatch!'

                scan = spectrum['params'].get('scan', None)
                if(scan==None):
                    scan = spectrum['params'].get('scans', None)

                charge = int(spectrum['params']['charge'][0])

                pepmass = spectrum['params']['pepmass'][0]
                precursor_mass = pepmass * charge - self._meta.proton * charge

                peaks=[]
                for mz, it in zip(mz_array, it_array):
                    peaks.append(str(mz)+':'+str(it))
                peaks=','.join(peaks)
                output_line = [str(scan), tokenized_seq, str(precursor_mass), str(charge), '-', peaks]
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

                precursor_mass = pepmass * charge - self._meta.proton * charge

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
