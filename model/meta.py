import os
import sys
import re
import numpy as np
import itertools
import pprint as pp
import xml.etree.ElementTree as ET
from .utils import UTILS
from typing import Dict, Iterable, Optional, Union

from spectrum_utils import fragment_annotation as fa, proforma, utils

'''
from depthcharge.data import (
    AnnotatedSpectrumDataset,
    CustomField,
    SpectrumDataset,
    preprocessing,
)
'''

class META:
    def __init__(self, configs):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(current_dir, 'unimod')
        unimod_xml = os.path.join(folder_path, 'unimod.xml')
        ptm_list = os.path.join(folder_path, 'PTMs.list')
        self._utils = UTILS()

        #configs
        self._configs = configs

        #ptms
        self._ptm_dict = self._read_ptms(ptm_list)

        #unimod
        self._elements, self._residues, self._modifications = self._parse_unimod(unimod_xml)

        #mass
        self._proton = self._elements['H'] - self._elements['e']
        self._hydroxide_ion = self._elements['H'] + self._elements['O'] + self._elements['e']

        #tokens
        self._tokens, self._special_tokens, self._mass_dict = self._make_tokens()

        #Onehot
        (self._onehot_to_residue, self._residue_to_onehot)=self._make_onehot_table(self._tokens)
        

        '''
        print('tokens',end=':')
        print(len(self._tokens))
        pp.pprint(self._tokens)
        print('-'*100)

        print('special_tokens',end=':')
        print(len(self._special_tokens))
        pp.pprint(self._special_tokens)
        print('-'*100)
       
        print('proton:')
        print(self._proton)
        print('-'*100)

        print('self._hydroxide_ion')
        print(self._hydroxide_ion)
        print('-'*100)

        print('elements:')
        pp.pprint(self._elements)
        print('-'*100)

        print('residues:')
        pp.pprint(self._residues)
        print('-'*100)

        print('ptm_dict')
        pp.pprint(self._ptm_dict)
        print('-'*100)

        print('modifications:')
        pp.pprint(self._modifications)
        print('-'*100)
        '''

        self._min_mz = self._configs['Model']['Spectrum']['min_mz']
        self._max_mz = self._configs['Model']['Spectrum']['max_mz']

        self._min_peaks = self._configs['Model']['Spectrum']['min_peaks']
        self._max_peaks = self._configs['Model']['Spectrum']['max_peaks']

        self._min_intensity = self._configs['Model']['Spectrum']['min_intensity']
        self._max_charge = self._configs['Model']['Spectrum']['max_charge']

        self._remove_precursor_tol = self._configs['Model']['Spectrum']['remove_precursor_tol']

    @property
    def configs(self):
        return(self._configs)

    @property
    def min_mz(self):
        return(self._min_mz)

    @property
    def max_mz(self):
        return(self._max_mz)

    @property
    def max_peaks(self):
        return(self._max_peaks)

    @property
    def min_peaks(self):
        return (self._min_peaks)

    @property
    def max_charge(self):
        return (self._max_charge)

    @property
    def elements(self):
        return(self._elements)

    @property
    def proton(self):
        return(self._proton)

    @property
    def empty_residue(self):
        return(self._special_tokens['-'])

    @property
    def N_term(self):
        return(self._special_tokens['N-term'])
 
    @property
    def C_term(self):
        return(self._special_tokens['C-term'])

    @property
    def residues(self):
        return(self._residues)

    @property
    def tokens(self):
        return(self._tokens)

    @property
    def special_tokens(self):
        return(self._special_tokens)

    @property
    def mass_dict(self):
        return(self._mass_dict)
    
    @property
    def onehot_to_residue(self):
        return(self._onehot_to_residue)

    @property
    def residue_to_onehot(self):
        return(self._residue_to_onehot)

    @property
    def sorted_peptides_mass_arr(self):
        return(self._make_residue_combination_pool())

    def _read_ptms(self, input_file):
        ptm_dict = dict()
        f_in=open(input_file,'r')
        for line in f_in:
            line=line.strip()
            if(line == ''):
                continue
            m=re.search('^#',line)
            if(m):
                continue
            arr=line.split('\t')
            residue = arr[0]
            ptm = arr[1]
            ptm_type = arr[2]
            if(ptm_type not in ptm_dict):
                ptm_dict[ptm_type] = dict()
            if(ptm not in ptm_dict[ptm_type]):
                ptm_dict[ptm_type][ptm] = set()
            ptm_dict[ptm_type][ptm].add(residue)
        f_in.close()
        return(ptm_dict)
        
    def _parse_unimod(self, unimod_xml):
        elements = dict()
        residues = dict()
        modifications = dict()       

        tree = ET.parse(unimod_xml)
        root = tree.getroot()
       
        namespace = root.tag.split('}')[0].strip('{')  # Extracts the namespace URI
        ns = {'umod': namespace}  # Map it to a prefix for use in queries

        umod_elements = root.find('umod:elements', ns)
        if(umod_elements is not None):
            for elem in umod_elements.findall('umod:elem', ns):
                title = elem.get('title')
                full_name = elem.get('full_name')
                avge_mass = elem.get('avge_mass')
                mono_mass = elem.get('mono_mass', None)
                elements[title] = float(mono_mass)

        umod_amino_acids = root.find('umod:amino_acids', ns)
        if(umod_amino_acids is not None):
            for aa in umod_amino_acids.findall('umod:aa', ns):
                title = aa.get('title')
                mono_mass = aa.get('mono_mass', None)
                if(not mono_mass):
                    sys.exit('Error:check mono_mass of aa')
                residues[title] = float(mono_mass)

        umod_modifications = root.find('umod:modifications', ns)
        if(umod_modifications is not None):
            for mod in umod_modifications.findall('umod:mod', ns):
                modi = mod.get('title', None)
                if(not modi):
                    sys.exit('Error:check title of mod!')
                if(modi not in modifications):
                    modifications[modi] = [None, set()]
                    
                for specificity in mod.findall('umod:specificity', ns):
                    site = specificity.get('site', None)
                    modifications[modi][1].add(site)

                for delta in mod.findall('umod:delta', ns):
                    mono_mass = delta.get('mono_mass', None)
                    modifications[modi][0] = float(mono_mass)

        elements['*'] = 0.0 #null element, aka nothing
        #elements['-'] = elements['*'] #alias of null element

        return(elements, residues, modifications)

    def _build_token(self,tokens, ptms, fixed=False):
        remove_residues = set()

        for ptm in ptms:
            residues = ptms[ptm]
            ptm_mass = sum([self._modifications[p][0] for p in ptm.split('+')])

            for p in ptm.split('+'):
                if(p not in self._modifications):
                    sys.exit('Error: The ptm '+p+' is not in unimod modifications, please check!')

            for r in residues:
                if(r in {'<', 'N-term', '>', 'C-term'}):        # 用集合 O(1) 平均时间复杂度
                    t = ptm
                    m = ptm_mass
                else:
                    t = r+'+'+ptm
                    m = tokens[r] + ptm_mass

                tokens[t]=m
                if(fixed):
                    remove_residues.add(r)

        return(remove_residues)

    def _make_tokens(self):
        tokens = dict()
        special_tokens = dict()

        if(self._residues['I'] != self._residues['L']):
            sys.exit('Error: residue I and L must have the same mass!')

        special_tokens['I'] = self._residues.pop('I')
        special_tokens['L'] = self._residues.pop('L')
        special_tokens['U'] = self._residues.pop('U')
        tokens['X'] = special_tokens['I']

        special_tokens['-'] = self._residues.pop('-') #pad token, aka nothing

        while self._residues:
            key, value = self._residues.popitem()
            tokens[key] = value

        variable_ptms = self._ptm_dict['variable']
        self._build_token(tokens, variable_ptms)

        fixed_ptms = self._ptm_dict['fixed']
        remove_residues = self._build_token(tokens, fixed_ptms, True)

        for r in remove_residues.union({'<', 'N-term', '>', 'C-term'}):
            t=tokens.pop(r, None)
            if(t != None):
                special_tokens[r]=t

        if(set(tokens.keys()) & set(special_tokens.keys())):
            sys.exit("tokens and special_tokens should not have common keys!")

        mass_dict = dict()
        mass_dict.update(tokens)
        mass_dict.update(special_tokens)

        return(tokens, special_tokens, mass_dict)

    def _make_onehot_table(self, dict_):
        onehot_to = list(dict_.keys())
        to_onehot = dict()
        for k,v in enumerate(onehot_to):
             to_onehot[v]=k
        return(onehot_to, to_onehot)

    def _make_residue_combination_pool(self):
        residue_mass_values = list(self._tokens.values())

        #Probes
        probe_layers = int(self._configs['MCTTS']['Tree']['probe_layers'])
        ceiling = int(self._configs['MCTTS']['Delta']['ceiling'])
    
        peptides_mass_arr = [0.0]
        for i in range(1, ceiling+1):
            combi=itertools.combinations_with_replacement(residue_mass_values, i)
            combi=list(combi)
            mass_arr = [sum(x) for x in combi]
            peptides_mass_arr.extend(mass_arr)

            if(i <= probe_layers*1):
                mass_arr_negative = [-x for x in mass_arr]
                peptides_mass_arr.extend(mass_arr_negative)
                #print(mass_arr_negative)

        return(sorted(peptides_mass_arr))

    def preprocess_spectrum(self, spectrum):
        # Spectrum preprocessing functions.
        self.spectrum = spectrum
        #self._utils.parse_var(self.spectrum['m/z array'])
        #self._utils.parse_var(self.spectrum['intensity array'])
        self.set_mz_range(min_mz=self._min_mz, max_mz=self._max_mz)
        self.remove_precursor_peak(fragment_tol_mass=self._remove_precursor_tol, fragment_tol_mode='Da')
        self.scale_intensity(scaling='root', max_intensity=1)
        self.filter_intensity(min_intensity=self._min_intensity, max_num_peaks=self._max_peaks)
        self.discard_low_quality(min_peaks=self._min_peaks)
        self.scale_to_unit_norm()
        return(self.spectrum)

    def set_mz_range(self, min_mz: Optional[float] = None, max_mz: Optional[float] = None):
        mz = self.spectrum['m/z array']
        intensity = self.spectrum['intensity array']
        if not (min_mz is None and max_mz is None):
            if min_mz is None:
                min_mz = mz[0]
            if max_mz is None:
                max_mz = mz[-1]
            if max_mz < min_mz:
                min_mz, max_mz = max_mz, min_mz
        min_i, max_i = 0, len(mz)
        while min_i < len(mz) and mz[min_i] < min_mz:
            min_i += 1
        while max_i > 0 and mz[max_i - 1] > max_mz:
            max_i -= 1
        self.spectrum['m/z array'] = mz[min_i:max_i]
        self.spectrum['intensity array'] = intensity[min_i:max_i]

    def remove_precursor_peak(self,
          fragment_tol_mass: float, #
          fragment_tol_mode: str, #fragment mass tolerance unit, "Da" or "ppm".这是这是
          isotope: int = 0,
        ):

        adduct_mass = self._proton

        charge = int(self.spectrum['params']['charge'][0])
        precursor_mz = self.spectrum['params']['pepmass'][0]
        neutral_mass = (precursor_mz - adduct_mass) * charge

        mz = self.spectrum['m/z array']
        intensity = self.spectrum['intensity array']

        c_mass_diff = 1.003355
        remove_mz = [
            (neutral_mass + iso * c_mass_diff) / charge + adduct_mass
            for charge in range(charge, 0, -1)
            for iso in range(isotope + 1)
        ]

        mask = np.full_like(mz, True, np.bool_)
        mz_i = remove_i = 0

        while mz_i < len(mz) and remove_i < len(remove_mz):
            md = utils.mass_diff(mz[mz_i], remove_mz[remove_i], fragment_tol_mode==fragment_tol_mode)
            if md < -fragment_tol_mass:
                mz_i += 1
            elif md > fragment_tol_mass:
                remove_i += 1
            else:
                mask[mz_i] = False
                mz_i += 1

        mz[:] = [mz[i] for i, x in enumerate(mask) if x]
        intensity[:] = [intensity[i] for i, x in enumerate(mask) if x]

        self.spectrum['m/z array'] = mz
        self.spectrum['intensity array'] = intensity

    def scale_intensity(self,
            scaling: Optional[str] = None,
            max_intensity: Optional[float] = None,
            degree: int = 2,
            base: int = 2,
            max_rank: Optional[int] = None,
        ):
        intensity = self.spectrum['intensity array']

        if scaling == "root":
            intensity = np.power(intensity, 1 / degree).astype(np.float32)
        elif scaling == "log":
            intensity = (np.log1p(intensity) / np.log(base)).astype(np.float32)
        elif scaling == "rank":
            if max_rank is None:
                max_rank = len(intensity)
            if max_rank < len(intensity):
                raise ValueError(
                    "`max_rank` should be greater than or equal to the number "
                    "of peaks in the spectrum. See `filter_intensity` to "
                    "reduce the number of peaks in the spectrum."
                )
            intensity = (max_rank - np.argsort(np.argsort(intensity)[::-1])).astype(np.float32)

        if max_intensity is not None:
            intensity = (intensity * max_intensity / intensity.max()).astype(np.float32)

        self.spectrum['intensity array'] = intensity.tolist()

    def filter_intensity(self, min_intensity: float = 0.0, max_num_peaks: Optional[int] = None):
        mz = self.spectrum['m/z array']
        intensity = self.spectrum['intensity array']
        if max_num_peaks is None:
            max_num_peaks = len(intensity)
        intensity_idx = np.argsort(intensity)
        min_intensity *= intensity[intensity_idx[-1]]
        # Discard low-intensity noise peaks.
        start_i = 0
        for idx in intensity_idx.tolist():
            intens = intensity[idx]
            if intens > min_intensity:
                break
            start_i += 1
        # Only retain at most the `max_num_peaks` most intense peaks.
        mask = np.full_like(intensity, False, np.bool_)
        mask[ intensity_idx[max(start_i, len(intensity_idx) - max_num_peaks) :] ] = True
        mz[:] = [mz[i] for i, x in enumerate(mask) if x]
        intensity[:] = [intensity[i] for i, x in enumerate(mask) if x]
        self.spectrum['m/z array'] = mz
        self.spectrum['intensity array'] = intensity

    def discard_low_quality(self, min_peaks: int):
        mz = self.spectrum['m/z array']
        if len(mz) < min_peaks:
            self.spectrum = None

    def scale_to_unit_norm(self):
        if(self.spectrum is not None):
            intensity = self.spectrum['intensity array']
            intensity = intensity / np.linalg.norm(intensity)
            self.spectrum['intensity array'] = intensity.tolist()