import os
import sys
import re
import itertools
import pprint as pp
import xml.etree.ElementTree as ET
from .utils import UTILS

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
    @property
    def configs(self):
        return(self._configs)

    @property
    def min_mz(self):
        return(self._configs['Model']['Basic']['min_mz'])

    @property
    def max_mz(self):
        return(self._configs['Model']['Basic']['max_mz'])

    @property
    def max_peaks(self):
        return self._configs['Model']['Basic']['max_peaks']

    @property
    def min_charges(self):
        return self._configs['Model']['Basic']['min_charges']

    @property
    def max_charges(self):
        return self._configs['Model']['Basic']['max_charges']

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
