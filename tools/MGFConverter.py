import os
import sys
import re
import pprint as pp
import numpy as np
import h5py, hdf5plugin
import random

import xmltodict
import json

from matchms.importing import load_from_msp
import pyopenms


import ast
import operator

from decimal import Decimal, ROUND_HALF_UP

from pathlib import Path


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
        self._proton = self._meta.proton
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
            if(self._input_format=='MassIVE_KB'):
                token = nums_str
            elif(self._input_format=='9species'):
                token = ''.join(mods)
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
        elif (self._output_format == 'InstaNovo'):
            if (nums_size > 1):
                nums = self.eval_with_precision_policy(nums_str)
            else:
                nums = mods[0]
            token = '[' + nums + ']'

            if (name == '<'):
                token = token
            else:
                token = name + token
        else:
            sys.exit(f'The method of replacing mass with tokens must be specified for the output_format {self._output_format}')

        #if (False and nums_size > 1):
        if (True):
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

    def batch_write_to_MGF(self, input_mgf, output_mgf, remove_charge_sign=True):
        spectra_buffer = []
        batch_size = 100  # 每100个谱图写入一次

        first_batch = True
        with MGF(input_mgf) as reader:
            mode = 'w'
            for spectrum in reader:
                SCANS = spectrum['params']['scans']
                if((self.scan_table is not None) and (SCANS not in self.scan_table)):
                    continue

                seq = spectrum['params']['seq']
                tokenized_seq, max_ptm_on_one_residue = self.modify_seq_to_format(seq)
                if ((tokenized_seq is None) or (max_ptm_on_one_residue is None)):
                    continue
                if (max_ptm_on_one_residue > self._allowed_max_ptm_on_one_residue):
                    continue
                PEPMASS = spectrum['params']['pepmass'][0]
                spectrum['params']['pepmass'] = PEPMASS

                if(self._output_format=='Casanovo'):
                    spectrum['params']['seq'] = tokenized_seq
                    #spectrum['params']['pepmass'] = PEPMASS
                elif(self._output_format=='InstaNovo'):
                    spectrum['params']['seq'] = tokenized_seq
                elif(self._output_format=='PrimeNovo'):
                    #PEPMASS = spectrum['params']['pepmass'][0]
                    CHARGE = spectrum['params']['charge']
                    SEQ = tokenized_seq
                    if(self._input_format == 'MassIVE_KB'):
                        TITLE = spectrum['params']['provenance_filename'] + ',' + spectrum['params']['provenance_scan']
                        RTINSECONDS = 0.0
                    if(self._input_format == '9species'):
                        TITLE = spectrum['params']['title']
                        RTINSECONDS = spectrum['params']['rtinseconds']

                    spectrum['params']={
                        'title':TITLE,
                        'pepmass':PEPMASS,
                        'charge':CHARGE,
                        'scans':SCANS,
                        'seq':SEQ,
                        'rtinseconds':RTINSECONDS
                    }
                else:
                    raise ValueError('Unknown formats')

                spectra_buffer.append(spectrum)

                # 批量写入以减少内存使用
                if len(spectra_buffer) >= batch_size:
                    mode = 'w' if first_batch else 'a'
                    with open(output_mgf, mode) as f:
                        if(remove_charge_sign):
                            #mgf.write(spectra_buffer, f, param_formatters={'charge': self.charge_formatter})
                            mgf.write(spectra_buffer, f, key_order = ['title', 'pepmass', 'charge', 'scans', 'rtinseconds'],
                                      param_formatters={'charge': self.charge_formatter})
                        else:
                            mgf.write(spectra_buffer, f, key_order = ['title', 'pepmass', 'charge', 'scans', 'rtinseconds'])
                    spectra_buffer = []
                    first_batch = False

            # 写入剩余的谱图
            if spectra_buffer:
                with open(output_mgf, mode) as f:
                    if (remove_charge_sign):
                        mgf.write(spectra_buffer, f, key_order = ['title', 'pepmass', 'charge', 'scans', 'rtinseconds'],
                                  param_formatters={'charge': self.charge_formatter})
                    else:
                        mgf.write(spectra_buffer, f, key_order = ['title', 'pepmass', 'charge', 'scans', 'rtinseconds'])

    def convert_MassiveKB_to_InstaNovo(self, mgf_file, output_prefix=None, dryrun=False):
        if(output_prefix is None):
            base_path = Path(mgf_file)
        else:
            base_path = Path(output_prefix)
        InstaNovo_mgf_file = base_path.with_suffix(base_path.suffix + ".InstaNovo.mgf")

        if(not dryrun):
            self.batch_write_to_MGF(input_mgf=mgf_file, output_mgf=InstaNovo_mgf_file, remove_charge_sign=False)

        return(InstaNovo_mgf_file)

    def convert_MassiveKB_to_PrimeNovo(self, mgf_file, PrimeNovo_mgf_file, remove_charge_sign=False, dryrun=False):
        print('raw_mgf',end=':\t')
        print(mgf_file)
        print('PrimeNovo_mgf',end=':\t')
        print(PrimeNovo_mgf_file)
        if(not dryrun):
            self.batch_write_to_MGF(input_mgf=mgf_file, output_mgf=PrimeNovo_mgf_file, remove_charge_sign=remove_charge_sign)

        return(PrimeNovo_mgf_file)

    def charge_formatter(self, param_name, param_value):
        if param_name.lower() == 'charge':
            if hasattr(param_value, 'int') and param_value:
                return 'CHARGE='+str(param_value.int)
            elif hasattr(param_value, '__len__') and len(param_value) > 0:
                try:
                    return 'CHARGE='+str(int(param_value[0]))
                except:
                    return 'CHARGE='+str(param_value)
        return 'CHARGE='+str(param_value)

    def convert_MassiveKB_to_PointNovo(self, mgf_file, output_prefix=None, dryrun=False):
        if(output_prefix):
            base_path = Path(output_prefix)
        else:
            base_path = Path(mgf_file)
        PointNovo_mgf_file = base_path.with_suffix(base_path.suffix + ".PointNovo.mgf")
        PointNovo_csv_file = base_path.with_suffix(base_path.suffix + ".PointNovo.csv")

        print('raw_mgf',end=':\t')
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
                    SCANS = spectrum['params']['scans']
                    if ((self.scan_table is not None) and (SCANS not in self.scan_table)):
                        continue

                    seq = spectrum['params']['seq']
                    tokenized_seq, max_ptm_on_one_residue = self.modify_seq_to_format(seq)
                    if ((tokenized_seq is None) or (max_ptm_on_one_residue is None)):
                        continue
                    if (max_ptm_on_one_residue > self._allowed_max_ptm_on_one_residue):
                        continue

                    TITLE = spectrum['params']['provenance_filename'] + ',' + spectrum['params']['provenance_scan']
                    PEPMASS = spectrum['params']['pepmass'][0]
                    RTINSECONDS = 0.0

                    CHARGE = spectrum['params']['charge']
                    if(CHARGE and (len(CHARGE) == 1)):
                        CHARGE = int(CHARGE[0])
                    else:
                        raise ValueError('The precursor charge is missing!')

                    spec_group_id = SCANS
                    mz = str(PEPMASS)
                    rt_mean = str(RTINSECONDS)
                    feature_area = '10.0'
                    irt = '0'
                    profile = str(rt_mean) + ':' + str(feature_area)

                    csv_arr = [spec_group_id, mz, str(CHARGE), rt_mean, tokenized_seq, SCANS, profile, feature_area, irt]
                    f_out_f1.write(','.join(csv_arr)+'\n')

                    spectrum['params']={
                        'title':TITLE,
                        'pepmass':PEPMASS,
                        'charge':CHARGE,
                        'scans':SCANS,
                        'rtinseconds': RTINSECONDS
                    }

                    spectra_buffer.append(spectrum)

                    # 批量写入以减少内存使用
                    if len(spectra_buffer) >= batch_size:
                        mode = 'w' if first_batch else 'a'
                        with open(PointNovo_mgf_file, mode) as f:
                            mgf.write(spectra_buffer, f, key_order=['title', 'pepmass', 'charge','scans' ,'rtinseconds'],
                                      param_formatters={'charge': self.charge_formatter})
                        spectra_buffer = []
                        first_batch = False

                # 写入剩余的谱图
                if spectra_buffer:
                    with open(PointNovo_mgf_file, mode) as f:
                        mgf.write(spectra_buffer, f, key_order=['title', 'pepmass', 'charge','scans' ,'rtinseconds'],
                                  param_formatters={'charge': self.charge_formatter})

            f_out_s1.close()
            f_out_f1.close()

        return(PointNovo_mgf_file, PointNovo_csv_file)

    def convert_9species_to_PointNovo(self, mgf_file, output_prefix=None, dryrun=False):
        if(output_prefix):
            base_path = Path(output_prefix)
        else:
            base_path = Path(mgf_file)
        PointNovo_mgf_file = base_path.with_suffix(base_path.suffix + ".PointNovo.mgf")
        PointNovo_csv_file = base_path.with_suffix(base_path.suffix + ".PointNovo.csv")

        print('raw_mgf',end=':\t')
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
            with (MGF(mgf_file) as reader):
                mode = 'w'
                for spectrum in reader:
                    SCANS = spectrum['params']['scans']
                    if ((self.scan_table is not None) and (SCANS not in self.scan_table)):
                        continue

                    seq = spectrum['params']['seq']
                    tokenized_seq, max_ptm_on_one_residue = self.modify_seq_to_format(seq)
                    if ((tokenized_seq is None) or (max_ptm_on_one_residue is None)):
                        continue
                    if (max_ptm_on_one_residue > self._allowed_max_ptm_on_one_residue):
                        continue

                    TITLE = spectrum['params']['title']
                    PEPMASS = spectrum['params']['pepmass'][0]
                    RTINSECONDS = spectrum['params']['rtinseconds']
                    CHARGE = spectrum['params']['charge']



                    if(CHARGE and (len(CHARGE) == 1)):
                        CHARGE = int(CHARGE[0])
                    else:
                        raise ValueError('The precursor charge is missing!')

                    #sys.exit()

                    spec_group_id = SCANS
                    mz = str(PEPMASS)
                    rt_mean = str(RTINSECONDS)
                    feature_area = '10.0'
                    irt = '0'
                    profile = str(rt_mean) + ':' + str(feature_area)

                    csv_arr = [spec_group_id, mz, str(CHARGE), rt_mean, tokenized_seq, SCANS, profile, feature_area, irt]
                    f_out_f1.write(','.join(csv_arr)+'\n')

                    spectrum['params']={
                        'title':TITLE,
                        'pepmass':PEPMASS,
                        'charge':CHARGE,
                        'scans':SCANS,
                        'rtinseconds': RTINSECONDS
                    }

                    spectra_buffer.append(spectrum)

                    # 批量写入以减少内存使用
                    if len(spectra_buffer) >= batch_size:
                        mode = 'w' if first_batch else 'a'
                        with open(PointNovo_mgf_file, mode) as f:
                            mgf.write(spectra_buffer, f, key_order=['title', 'pepmass', 'charge','scans' ,'rtinseconds'],
                                      param_formatters={'charge': self.charge_formatter})
                        spectra_buffer = []
                        first_batch = False

                # 写入剩余的谱图
                if spectra_buffer:
                    with open(PointNovo_mgf_file, mode) as f:
                        mgf.write(spectra_buffer, f, key_order=['title', 'pepmass', 'charge','scans' ,'rtinseconds'],
                                  param_formatters={'charge': self.charge_formatter})

            f_out_s1.close()
            f_out_f1.close()

        return(PointNovo_mgf_file, PointNovo_csv_file)

    def convert_spec_to_PointNovo(self, spec_file, output_prefix=None, dryrun=False):
        if(not output_prefix):
            output_prefix = spec_file
        base_path = Path(output_prefix)
        bn = os.path.basename(spec_file)

        PointNovo_mgf_file = base_path.with_suffix(base_path.suffix + ".PointNovo.mgf")
        PointNovo_csv_file = base_path.with_suffix(base_path.suffix + ".PointNovo.csv")

        print('spec_file',end=':\t')
        print(spec_file)
        print('PointNovo_mgf',end=':\t')
        print(PointNovo_mgf_file)
        print('PointNovo_csv',end=':\t')
        print(PointNovo_csv_file)

        if (not dryrun):
            f_out_f1 = open(PointNovo_csv_file, 'w')
            f_out_f1.write('spec_group_id,m/z,z,rt_mean,seq,scans,profile,feature area,irt\n')

            first_batch=True
            f_in = open(spec_file, 'r')
            for line in f_in:
                line=line.strip()
                m=re.search('^#',line)
                if(m or line==''):
                    continue
                arr = line.split('\t')

                scans = arr[0]
                tokenized_seq = self.replace_ptm(arr[1])
                mass=float(arr[2])
                charge=int(arr[3])
                irt = arr[4]
                mz = (mass/charge) + self._proton

                spec_group_id = scans
                #rt_mean = irt
                rt_mean = 800
                feature_area = '10.0'
                profile = str(rt_mean) + ':' + str(feature_area)

                csv_arr = [spec_group_id, str(mz), str(charge), str(rt_mean), tokenized_seq, scans, profile, feature_area, irt]
                f_out_f1.write(','.join(csv_arr) + '\n')

                spectrum=dict()
                spectrum['params'] = {
                    'title': bn+':'+scans,
                    'pepmass': mz,
                    'charge': charge,
                    'scans': scans,
                    'rtinseconds': irt,
                }

                spectrum['m/z array'] = []
                spectrum['intensity array'] = []
                peaks = arr[-1].split(',')
                for pair in peaks:
                    mz, intensity = pair.split(':')
                    spectrum['m/z array'].append(mz)
                    spectrum['intensity array'].append(intensity)

                mode = 'w' if first_batch else 'a'
                with open(PointNovo_mgf_file, mode) as f:
                    mgf.write([spectrum], f, key_order=['title', 'pepmass', 'charge', 'scans', 'rtinseconds'],
                              param_formatters={'charge': self.charge_formatter})
                first_batch = False

            f_in.close()
            f_out_f1.close()

    def convert_MassiveKB_to_CasanovoMGF(self, mgf_file, casanovomgf_file=None, dryrun=False):
        if(not casanovomgf_file):
            casanovomgf_file=mgf_file+'.casanovo.mgf'
        if(dryrun):
            return(casanovomgf_file)
        self.batch_write_to_MGF(input_mgf=mgf_file, output_mgf=casanovomgf_file)
        return(casanovomgf_file)

    def convert_9species_to_MGF(self, input_mgf_file, output_mgf_file=None, dryrun=False):
        if(not output_mgf_file):
            output_mgf_file=input_mgf_file+'.'+ self._output_format +'.mgf'
        if(dryrun):
            return(output_mgf_file)

        self.batch_write_to_MGF(input_mgf=input_mgf_file, output_mgf=output_mgf_file)
        return (output_mgf_file)

    def convert_9species_to_PrimeNovo(self, input_mgf_file, output_mgf_file=None, remove_charge_sign=False, dryrun=False):
        if(not output_mgf_file):
            output_mgf_file=input_mgf_file+'.PrimeNovo.mgf'
        if(dryrun):
            return(output_mgf_file)

        self.batch_write_to_MGF(input_mgf=input_mgf_file, output_mgf=output_mgf_file, remove_charge_sign=remove_charge_sign)
        return(output_mgf_file)

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

    def convert_MassiveKB_to_PepGo(self, mgf_file, spec_file=None, dryrun=False, preprocess=False):
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
        return(z)

    def convert_msp_to_spec(self, input_file, output_file=None):
        def write_row(row, f):
            if row:
                Scan = row.get('Scan')
                Peptide = row.get('Peptide')
                Mass = row.get('Mass')
                Charge = row.get('Charge')
                iRT = row.get('iRT')
                Peaks = row.get('Peaks')
                Num_peaks = row.get('Num_peaks')
                Collision = row.get('Collision')

                if Num_peaks != len(Peaks):
                    ppp = Peptide.replace(',', '')
                    nnn = len(Peaks)
                    raise ValueError(f'Number of actual peaks({nnn}) does not match Num_peaks({Num_peaks}) in peptide {ppp}')

                output_row = '\t'.join([Scan, Peptide, Mass, Charge, iRT, Collision, str(Num_peaks), ','.join(Peaks)])
                f.write(output_row + '\n')
                row.clear()
            return row

        def peptide_to_seqarr(peptide, Mods_num, Mods):
            side_chain = [''] * len(peptide)
            if (Mods_num):
                for mod in Mods.split('/'):
                    (pos, residue, ptm) = mod.split(',')
                    pos = int(pos)
                    side_chain[pos] = ptm

            seqarr = []
            for i, r in enumerate(peptide):
                if (self._replace_isoleucine_and_leucine_with_X and (r == 'I' or r == 'L')):
                    r = 'X'
                ptm = side_chain[i]
                if (ptm):
                    r = r + '+' + ptm
                seqarr.append(r)

            return (seqarr)

        bn=os.path.basename(input_file)
        if(output_file is None):
            output_file = bn+'.spec'
        f_out=open(output_file,'w')
        prefix=bn

        #f_out.write('#Peptide\tCharge\tMW\tMods_num\tMods\tiRT\tCollision\tID\tNum_peaks\tIons(mz:intensity)')
        #f_out.write('#Title\tPeptide\tMass\tCharge\tiRT\tMods_num\tMods\tCollision\tID\tNum_peaks\tIons(mz:intensity)')
        f_out.write('#Scan\tPeptide\tMass\tCharge\tiRT\tCollision\tNum_peaks\tPeaks(mz:intensity)'+'\n')

        bn_arr=bn.split('_')
        collision=bn_arr[1]

        total_peptide_num=0
        flag=0
        id_n=0
        spec_id='-'
        peptide='-'
        charge = None
        spec_row = dict()

        f_in=open(input_file, 'r')
        for line in f_in:
            line=line.strip()
            m=re.search('^#',line)
            if(m or line==''):
                continue

            m=re.search('^Name:',line)
            if(m):
                flag=0
                num_peaks=0
                total_peptide_num+=1
                name=line.replace("Name:",'')
                name=name.strip()
                m=re.search(r'(.+)/(\d+)$',name)

                write_row(spec_row, f_out)

                if(m):
                    peptide=m.group(1)
                    charge=m.group(2)
                    d = re.findall(r'n*\[(\d+)\]',peptide)
                    (peptide,subn) = re.subn(r'n*\[(\d+)\]','',peptide)
                    spec_id = prefix+':'+peptide+':'+str(id_n)
                    spec_row['peptide_without_ptm'] =  peptide
                    spec_row['Scan'] = spec_id
                    spec_row['Charge'] = charge

                    id_n+=1
                else:
                    peptide='-'
                    charge = None
                    spec_row['peptide_without_ptm'] = peptide
                    spec_row['Charge'] = charge
                    sys.exit('Name format error! ( Name:<peptide sequence>/<charge> )')
            else:
                if(flag==1):
                    m=re.search(r'^MW:\s*(\d+(\.\d+)?)$',line)
                    if(m):
                        mw=m.group(1)
                        spec_row['Mass'] = mw
                    else:
                        sys.exit('MW missing')
                elif(flag==2):
                    comment=line
                    m=re.search(r'\s+iRT=(\S+)$', comment)
                    if(m):
                        spec_row['iRT'] = m.group(1)
                    else:
                        sys.exit('iRT missing')

                    m=re.search(r'\s+Mods=(\d+)(/(\S+))?\s', comment)
                    if(m):
                        Mods_num = int(m.group(1))
                        if(Mods_num):
                            Mods = m.group(3)
                        else:
                            Mods='-'
                        seq_arr = peptide_to_seqarr(spec_row.get('peptide_without_ptm'), Mods_num, Mods)
                        spec_row['Peptide'] = ','.join(seq_arr)
                    else:
                        sys.exit('Mods missing')
                    spec_row['Collision'] = collision
                elif(flag==3):
                    m=re.search(r'^Num peaks:\s+(\d+)$', line)
                    if(m):
                        num_peaks=int(m.group(1))
                        spec_row['Num_peaks'] = num_peaks
                elif(flag>3):
                    arr=line.strip().split('\t')
                    mz=arr[0]
                    intensity=arr[1]
                    if('Peaks' not in spec_row):
                        spec_row['Peaks'] = []
                    peak = mz+':'+intensity
                    spec_row['Peaks'].append(peak)
            flag+=1

        write_row(spec_row, f_out)

        f_in.close()
        f_out.close()
        total_peptide_num += 1
        return(total_peptide_num)

    def replace_ptm(self, seq):
        model_ptms = self._meta.configs['PTMs'][self._output_format]
        seq_arr=seq.strip().split(',')
        tmp_arr=seq.strip().split(',')
        for i,r in enumerate(seq_arr):
            r_arr = r.split('+')
            if(len(r_arr)>1):
                r_arr[1:] = [model_ptms[ptm] for ptm in r_arr[1:]]
                seq_arr[i]=''.join(r_arr)
        seq=''.join(seq_arr)
        if(len(tmp_arr)!=len(seq_arr)):
            raise ValueError('Length of seq_arr is abnormal after modifcation')
        return seq

    def convert_spec_to_mgf(self, spec_file, output_mgf_file=None, dryrun=False, remove_charge_sign=True):
        if(not output_mgf_file):
            base_path = Path(spec_file)
            output_mgf_file = base_path.with_suffix(base_path.suffix + '.' + self._output_format +'.mgf')

        print('spec_file',end=':\t')
        print(spec_file)
        print('output_mgf_file',end=':\t')
        print(output_mgf_file)

        if (not dryrun):
            spectra_buffer = []
            batch_size = 100  # 每100个谱图写入一次
            first_batch = True
            f_in = open(spec_file, 'r')
            for line in f_in:
                spectrum = dict()
                line=line.strip()
                m=re.search('^#',line)
                if(m or line==''):
                    continue
                arr = line.split('\t')

                if('params' not in spectrum):
                    spectrum['params']=dict()
                scans=arr[0]
                mass=float(arr[2])
                charge=int(arr[3])

                spectrum['params']['title'] = 'ProteomeTools:'+scans
                mz = (mass/charge) + self._proton
                spectrum['params']['pepmass'] = mz
                spectrum['params']['charge'] = charge
                spectrum['params']['scans'] = scans
                spectrum['params']['rinseconds'] = arr[4]
                spectrum['params']['seq'] = self.replace_ptm(arr[1])

                spectrum['m/z array'] = []
                spectrum['intensity array'] = []
                pairs = arr[-1].split(',')
                for pair in pairs:
                    mz, intensity = pair.split(':')
                    spectrum['m/z array'].append(mz)
                    spectrum['intensity array'].append(intensity)

                spectra_buffer.append(spectrum)

                # 批量写入以减少内存使用
                if len(spectra_buffer) >= batch_size:
                    mode = 'w' if first_batch else 'a'
                    with open(output_mgf_file, mode) as f:
                        if remove_charge_sign:
                            mgf.write(spectra_buffer, f, key_order=['title', 'pepmass', 'charge', 'scans', 'rtinseconds'],
                                      param_formatters={'charge': self.charge_formatter})
                        else:
                            mgf.write(spectra_buffer, f, key_order=['title', 'pepmass', 'charge', 'scans', 'rtinseconds'])
                    spectra_buffer = []
                    first_batch = False

            f_in.close()

            if spectra_buffer:
                mode = 'w' if first_batch else 'a'
                with open(output_mgf_file, mode) as f:
                    if remove_charge_sign:
                        mgf.write(spectra_buffer, f, key_order=['title', 'pepmass', 'charge', 'scans', 'rtinseconds'],
                                  param_formatters={'charge': self.charge_formatter})
                    else:
                        mgf.write(spectra_buffer, f, key_order=['title', 'pepmass', 'charge', 'scans', 'rtinseconds'])
                spectra_buffer = []
                first_batch = False

        return(output_mgf_file)

    def replace_IL_with_X_in_spec(self, input_spec_file, output_spec_file=None, dryrun=False):
        print(input_spec_file)
        print(output_spec_file)
        if(not output_spec_file):
            base_path = Path(input_spec_file)
            output_spec_file = base_path.with_suffix(base_path.suffix + '.' + self._output_format +'.X.spec')

        print('input_spec_file',end=':\t')
        print(input_spec_file)
        print('output_spec_file',end=':\t')
        print(output_spec_file)

        f_out = open(output_spec_file, 'w')
        if (not dryrun):
            f_in = open(input_spec_file, 'r')
            for line in f_in:
                line=line.strip()
                m=re.search('^#',line)
                if(m or line==''):
                    f_out.write(line+'\n')
                    continue
                arr = line.split('\t')

                seq = arr[1]
                seq_arr = seq.strip().split(',')
                for i, r in enumerate(seq_arr):
                    r_arr = r.split('+')
                    if (len(r_arr) > 1 and (r_arr[0]=='I' or r_arr[0]=='L')):
                        r_arr[0]='X'
                        seq_arr[i] = '+'.join(r_arr)
                    if(r=='I' or r=='L'):
                        seq_arr[i]='X'
                arr[1] = ','.join(seq_arr)
                line='\t'.join(arr)
                f_out.write(line + '\n')
            f_in.close()
        f_out.close()

        return(output_spec_file)

    def unique_spec(self, input_spec_file, output_spec_file=None, dryrun=False):
        if(not output_spec_file):
            base_path = Path(input_spec_file)
            output_spec_file = base_path.with_suffix('.' + self._output_format)

        print('input_spec_file',end=':\t')
        print(input_spec_file)
        print('output_spec_file',end=':\t')
        print(output_spec_file)

        f_out = open(output_spec_file, 'w')
        if (not dryrun):
            spec_lines=dict()
            f_in = open(input_spec_file, 'r')
            for line in f_in:
                line=line.strip()
                m=re.search('^#',line)
                if(m or line==''):
                    f_out.write(line+'\n')
                    continue
                arr = line.split('\t')
                seq = arr[1]
                if(seq not in spec_lines):
                    spec_lines[seq]=[]
                spec_lines[seq].append(line)
            f_in.close()

            #print('spec_lines')
            #pp.pprint(spec_lines)
            '''
            for i in spec_lines:
                print(i)
                if(len(spec_lines[i])>1):
                    pp.pprint(spec_lines[i])
                print('-'*100)
            '''
            keys = list(spec_lines.keys())
            #pp.pprint(keys[:5])
            random.shuffle(keys)
            #pp.pprint(keys[:5])
            #sys.exit()
            for k in keys:
                selected = random.choice(spec_lines[k])
                f_out.write(selected + '\n')
        f_out.close()

        return(output_spec_file)