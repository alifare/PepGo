import re
import sys
import math
import time

from itertools import product
import numpy as np
import cupy as cp
import einops
import torch

np.set_printoptions(suppress=True)

import pandas as pd
import pprint as pp

import gc
import collections

from scipy import sparse
from pympler import asizeof
from pympler import muppy, summary

from .utils import UTILS
from treelib import Node, Tree

import bisect

class myNode():
    def __init__(self, residue='', mass=0.0, layer=0, parent=None):
        self.visits = 1
        self.parent = parent
        self.children = []
        self.residue= residue
        self.layer = layer
        self.reward = 0.0
        self.delta = [None, None, None, 0.0] #[Transformer beam delta, Docking beam delta, Bisect delta, zero delta]
        self.docking = 0.0
        self.path_dockings = []
        self.mass = mass
        self.cumulative_mass = self.add_mass()
        self.sequence = parent.sequence+','+residue if(parent) else residue

    def add_mass(self):
        cumulative_mass = self.parent.cumulative_mass + self.mass if(self.parent) else self.mass
        return(cumulative_mass)    

    def is_terminal(self):
        return(self.layer<=0)
    
    def add_child(self, child):
        self.children.append(child)
    
    def update(self):
        self.visits+=1

    def fully_expanded(self, breadth):
        return(len(self.children) == breadth)


class MCTTS_Node(Node):
    def __init__(self, residue='', mass=0.0, layer=0, ground_truth=False, parent=None):
        super().__init__()
        self.visits = 1
        self.parent = parent
        self.children = []
        self.residue = residue
        self.sequence = []
        self.layer = layer
        self.reward = np.longdouble(0.0)
        self.docking = np.longdouble(0.0)
        self.mass = mass
        self.cumulative_mass = self.add_mass()
        self.ground_truth=ground_truth

    def add_mass(self):
        cumulative_mass = self.mass
        if(self.parent != None):
            cumulative_mass += self.parent.cumulative_mass
        return(cumulative_mass)

    def add_child(self, child):
        self.children.append(child)
    
    def update(self):
        self.visits+=1

    def fully_expanded(self, breadth):
        if(len(self.children) == breadth):
            return True
        return False    

class Monte_Carlo_Double_Root_Tree:
    def __init__(self, meta, configs, Transformer_N=None, Transformer_C=None):
        print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' started '+ '+'*100)
        self._utils = UTILS()
        self._meta = meta 
        self._configs = configs

        #Meta infomation
        self._proton = self._meta.proton
        self._elements = self._meta.elements
        self._mass_dict = self._meta.mass_dict

        self._tokens = self._meta.tokens
        self._tokens_keys = list(self._tokens.keys())
        self._tokens_values = list(self._tokens.values())
        '''
        self._utils.parse_var(self._tokens, 'self._tokens')
        self._utils.parse_var(self._tokens_keys, 'self._tokens_keys')
        self._utils.parse_var(self._tokens_values, 'self._tokens_values')
        '''
        self._sorted_peptides_mass_arr = self._meta.sorted_peptides_mass_arr

        #Configs
        self._budget = int(self._configs['MCTTS']['Tree']['budget'])

        self._depth = int(self._configs['MCTTS']['Tree']['depth'])
        self._depth_Transformer = int(self._configs['MCTTS']['Tree']['depth_Transformer'])
        self._depth_Transformer_beam = int(self._configs['MCTTS']['Tree']['depth_Transformer_beam'])
        self._probe_layers = int(self._configs['MCTTS']['Tree']['probe_layers'])
        
        self._ion_type_left = self.make_eval_dict(self._configs['Model']['Probe']['ion_type']['left'])
        self._ion_type_right = self.make_eval_dict(self._configs['Model']['Probe']['ion_type']['right'])
        self._neural_loss = self.make_eval_dict(self._configs['Model']['Probe']['neural_loss'])

        self._ceiling = int(self._configs['MCTTS']['Delta']['ceiling'])
        self._standard_deviation = float(self._configs['MCTTS']['Delta']['standard_deviation'])

        #self._SCALAR = math.sqrt(2.0)
        #self._SCALAR = 0.001 * math.sqrt(2.0)
        self._SCALAR = 0.001
        #self._SCALAR_Transformer = math.sqrt(2.0)
        self._SCALAR_Transformer = 0.001
        self._pep_len = self._configs['Model']['Basic']['pep_len']

        #Delta reward generated in simulation step
        self.probe_bisect_search_delta = self._configs['MCTTS']['Delta']['mode']['probe_bisect_search']
        self.transformer_bisect_search_delta = self._configs['MCTTS']['Delta']['mode']['transformer_bisect_search']
        self.transformer_beam_search_delta = self._configs['MCTTS']['Delta']['mode']['transformer_beam_search']

        #Transformer models
        self.Transformer_N = Transformer_N
        self.Transformer_C = Transformer_C

        #Others
        self._size_upper_limit = pow(len(self._tokens_values), self._ceiling)

        print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)

    def readin_spectrum(self, spectrum):
        spectrum_cupy = cp.array(spectrum)
        #spectrum_cupy = cp.expand_dims(spectrum_cupy, axis=1)
        mz = spectrum_cupy[:,:,0]
        mz = mz[mz.nonzero()]

        intensity = spectrum_cupy[:,:,1]
        return(mz)

    def readin_charge(self, charge_num=None):
        if(charge_num==None):
            charge_num = self._configs['Model']['Probe']['charge_num']
        elif(not isinstance(charge_num, int)):
            sys.exit('charge_num must be an integer!')
            
        charge_str = ','.join([str(i)+':+'+'H'*i+':'+str(i) for i in range(1, charge_num+1)])
        charge = self.make_eval_dict(charge_str)
        #self._charge = self.make_eval_dict(self._configs['Model']['Probe']['charge'])
        return(charge)

    def make_eval_dict(self, string):
        eval_dict = collections.OrderedDict()
        if(string != ''):
            for s in string.strip().split(','):
                (name, ops, charge)=s.split(':')
                ops_string=''
                for ss in ops.split('&'):
                    op=ss[0]
                    elements_mass = [str(self._elements[i]) for i in list(ss[1:])]
                    ops_string += op+'('+'+'.join(elements_mass)+')'
                eval_dict[name]=[eval(ops_string), eval(charge)]
        return(eval_dict)

    def eval_str(self, number, ops):
        (summand, denominator)=ops
        s=(number + summand)/denominator
        return(s)

    def make_tensor_m(self, total_mass, A_mass, direction=None):
        M = A_mass

        if(direction=='left'):
            ion_type_first = self._ion_type_left
            ion_type_second = self._ion_type_right
        elif(direction=='right'):
            ion_type_first = self._ion_type_right
            ion_type_second = self._ion_type_left
        else:
            sys.exit('Direction must be specified as left or right!')

        #ion types
        arr=[]
        tokens=[]
        for i in ion_type_first:
            s = self.eval_str(M, ion_type_first[i])
            arr.append(s)
            tokens.append(i)

        B_mass = total_mass - M
        for i in ion_type_second:
            s = self.eval_str(B_mass, ion_type_second[i])
            arr.append(s)
            tokens.append(i)
        M = cp.stack(arr, axis=-1)

        #neural loss
        arr=[]
        token_arr=[]
        for i in self._neural_loss:
            s = self.eval_str(M, self._neural_loss[i])
            arr.append(s)
            token_arr.append(i)
        M = cp.stack(arr, axis=-1)
        tokens = ['-'.join(list(i)) for i in list(product(tokens, token_arr))]

        #charge
        arr=[]
        token_arr=[]
        for i in self._charge:
            s = self.eval_str(M, self._charge[i])
            arr.append(s)
            token_arr.append(i)
        tokens = ['-'.join(list(i)) for i in list(product(tokens, token_arr))]
        #tokens = ['-'.join(list(i)) for i in list(product(token_arr, tokens))]
        M = cp.stack(arr, axis=-1)
        M = cp.reshape(M, [M.shape[0], -1])

        return(M, tokens)

    def dock(self, probes):
        probes = cp.expand_dims(probes, axis=2)
        mz = cp.expand_dims(self._mz, axis=0)
        
        diff_raw = probes - mz

        diff_raw_reshape = cp.reshape(diff_raw, [diff_raw.shape[0],-1])
        diff_abs = cp.abs(diff_raw_reshape)

        diff_abs_argmin = cp.expand_dims(cp.argmin(diff_abs, axis=-1), axis=-1)
        diff_min = cp.take_along_axis(diff_raw_reshape, diff_abs_argmin, axis=-1)

        return(diff_min)

    def activation(self, docking):
        activ = math.exp( - pow(docking, 2)/(2 * pow(self._standard_deviation, 2)) )
        return(activ)

    def get_peptide_true(self, peptide):
        true = re.sub(',+\$+', '', ','.join(peptide)).replace('<,','').replace(',>','')
        return(true)

    '''
    def make_probes(self, steps=[0,1]):
        N_root = myNode(residue='<', mass=self._mass_dict['<'], layer= self._depth)
        C_root = myNode(residue='>', mass=self._mass_dict['>'], layer= self._depth)
        self._remaining_mass =  float(self._total_mass) - N_root.mass - C_root.mass - self._proton

        probes_info=[]
        for i in steps:
            pep_seq=N_root.residue
            for r in range(0, i):
                pep_seq+=self._true_peptide[r]
                N_root.cumulative_mass += self._mass_dict[self._true_peptide[r]]

            terminal_mass = cp.array(N_root.cumulative_mass, dtype=self._total_mass.dtype)
            probes, tokens = self.make_tensor_m(self._total_mass, terminal_mass, 'left')
            probes = probes.flatten()
            probes = probes.get()

            probes_info.append([self._true_peptide, pep_seq, probes, tokens, self._total_mass])
        return(probes_info)
    '''

    def plant_tree(self, tree, i, peptide_arr):
        for d in range(self._probe_layers):
            g = ''
            if(i<0):
                g = peptide_arr[i-d]
            else:
                g = peptide_arr[i+d]
            for leaf in tree.leaves():
                for k in self._tokens_keys:
                    nid = leaf.identifier+'_'+k
                    nmass = leaf.data.mass+self._mass_dict[k]
                    ntruth = (k==g and leaf.data.ground_truth)
                    n=Node(tag=k, identifier=nid, data=MCTTS_Node(mass=nmass, ground_truth=ntruth))
                    tree.add_node(n, parent=leaf)
        return(tree)

    def plant_tree2(self, tree):
        hierarchical_nodes = dict()
        for depth in range(1, self._probe_layers+1):
            for leaf in tree.leaves():
                for k in self._tokens_keys:
                    nid = leaf.identifier+'_'+k
                    nmass = leaf.data.mass+self._mass_dict[k]
                    ntruth = leaf.data.ground_truth
                    n=Node(tag=k, identifier=nid, data=MCTTS_Node(mass=nmass, layer=depth, ground_truth=ntruth))
                    tree.add_node(n, parent=leaf)
                    
                    if(depth >= self._probe_layers):
                        continue
                    if(depth not in hierarchical_nodes):
                        hierarchical_nodes[depth]=[]
                    hierarchical_nodes[depth].append(nid)

        return(tree, hierarchical_nodes)
    
    def make_branches(self, depth=None):
        print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' started '+ '+'*100)
       
        branches_arr=[]
        #for i in range(0, self._depth_Transformer):
        #for i in range(0, self._depth_Transformer+1):
        for i in range(0, depth + 1):
            branches_dict = dict()
            for e in product(self._tokens_keys, repeat=i):
                e = list(e)
                e_value = e
                #e_value = [self.model.decoder.tokenize_residue(j) for j in e]
                e_key = '_'.join(['root']+e)

                if(e_key not in branches_dict):
                    branches_dict[e_key] = e_value
                else:
                    sys.exit('Error: '+ e_key + 'already exists in branches_dict!')
            branches_arr.append(branches_dict)

        print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)
        return(branches_arr)

    def plant_perfect_tree(self, precursor, root, tail_mass, model, memories, mem_masks, mode, delta):
        print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' started '+ '+'*100)
        #mode=0 Transformer mode
        #mode=1 Spectrum Probes mode
        #self._utils.parse_var(precursor)

        root_residue_list = root.residue.split(',')
        root_residue_list_nohead = root_residue_list[1:]
        root_residue_list_key = '_'.join(root_residue_list)

        #self._utils.parse_var(root_residue_list)
        self._utils.parse_var(root_residue_list_nohead)
        #self._utils.parse_var(root_residue_list_key)
        #self._utils.parse_var(self.branches_arr, 'self.branches_arr')
        
        perfect_tree = dict()

        if(mode==1):
           pass 

        if(mode==0):
            for i in range(len(self.branches_arr)-1):
                print('loop:'+str(i))
                #self._utils.parse_var()
                branches_dict = self.branches_arr[i]
                branches_keys = list(branches_dict.keys())
                branches_values = list(branches_dict.values())

                self._utils.parse_var(branches_dict)
                self._utils.parse_var(branches_keys)
                self._utils.parse_var(branches_values)
                print('='*100)

                branches = []
                for e in branches_values:
                    e= root_residue_list_nohead + e
                    if(len(e)>=self._pep_len):
                        sys.exit('len(e)>=self._pep_len:'+len(e))
                    branches.append(e)

                #self._utils.parse_var(self._pep_len)

                for i, key in enumerate(branches_keys):
                    key = key.replace('root', root_residue_list_key)
                    for residue in model.residues.keys():
                        key_j = key +'_'+ residue
                        perfect_tree[key_j] = [None, None, None, None, None, 0.0]
                        #[Transformer reward, Docking reward, Transformer beam delta, Docking beam delta, Bisect delta, 0 delta]

                #self._utils.parse_var(perfect_tree, 'A')

                repeat_n = len(branches)
                batch_size = precursor.shape[0]

                if(i==0 and (root.residue == '<' or root.residue == '>')): #First prediction
                    branches=torch.zeros(batch_size, 0, dtype=torch.int64, device=model.decoder.device)
                    repeat_n=1
                    self._utils.parse_var(branches, 'A')
                else:
                    self._utils.parse_var(branches, 'B')
                    branches = model.tokenizer.tokenize(branches)
                    branches = branches.to(model.decoder.device)
                    self._utils.parse_var(branches, 'C')

                    #sys.exit('i=='+str(i))

                precursors_n = einops.repeat(precursor, "B L -> (B S) L", S=repeat_n)
                memories_n = einops.repeat(memories,  "B L V -> (S B) L V", S=repeat_n)
                mem_masks_n = einops.repeat(mem_masks, "B L -> (S B) L", S=repeat_n)
    
                precursors_n = precursors_n.to(model.decoder.device)
                memories_n = memories_n.to(model.decoder.device)
                mem_masks_n = mem_masks_n.to(model.decoder.device)

                self._utils.parse_var(branches, 'D')
                #print(memories_n.shape)
                #print(mem_masks_n.shape)
                #print(precursors_n.shape)


                #(logits, tokens) = model.decoder(
                logits = model.decoder(
                    tokens=branches,
                    memory=memories_n,
                    memory_key_padding_mask=mem_masks_n,
                    precursors=precursors_n
                )
                print('i==' + str(i))
                self._utils.parse_var(branches, 'E')

                self._utils.parse_var(logits)
                #self._utils.parse_var(perfect_tree, 'B')

                #(logits, tokens) = model.decoder(branches, precursors_n, memories_n, mem_masks_n, partial=True)

                last_logits = logits[:, -1, :]
                last_probs = torch.softmax(last_logits, dim=-1)
                last_prob_arr = last_probs.detach().cpu().numpy()
                #print('last_prob_arr.shape',end=':')
                #print(last_prob_arr.shape)

                for i, key in enumerate(branches_keys):
                    key = key.replace('root', root_residue_list_key)
                    #for j in range(1, last_prob_arr.shape[-1]-1):
                    for j in range(2, last_prob_arr.shape[-1]):
                        #print('j',end=':')
                        #print(j)
                        residue = model.tokenizer.detokenize_residue(j)
                        #print('')
                        #print('residue',end=':')
                        #print(residue)
                        key_j = key +'_'+ residue
                        perfect_tree[key_j][0] = last_prob_arr[i][j] #Transformer reward(probability)
                        #print('prob',end=':')
                        #print(perfect_tree[key_j][0])
                        #print('='*100)
                self._utils.parse_var(perfect_tree, 'C')
    
            #Beam search to get delta
            if(self.transformer_beam_search_delta and (delta == -4)):
                leaf_keys = list(self.branches_arr[-1].keys())
                leaves = list(self.branches_arr[-1].values())
                
                leaf_keys = [k.replace('root', root_residue_list_key) for k in leaf_keys]
                leaves = [root_residue_list_nohead+leaf for leaf in leaves]
    
                # Sizes.
                batch = len(leaves)  # B
                length = self._pep_len + 1  # L
                vocab = model.decoder.vocab_size + 1  # V
                beam = model.n_beams  # S
                print('B L V S:', batch, length, vocab, beam)
    
                # Initialize scores and tokens.
                scores = torch.full(
                    size=(batch, length, vocab, beam), fill_value=torch.nan
                )
                scores = scores.type_as(memories)
                
                tokens = torch.zeros(batch, length, beam, dtype=torch.int64)
                tokens = tokens.to(model.decoder.device)
    
                # Create cache for decoded beams.
                pred_cache = collections.OrderedDict((i, []) for i in range(batch))
    
                # Get the first prediction.
                precursors_n = einops.repeat(precursor, "N I -> (N B) I", B=batch)
                mem_masks_n = einops.repeat(mem_masks, "N I -> (N B) I", B=batch)
                memories_n = einops.repeat(memories,  "N I E -> (N B) I E", B=batch)
    
                precursors_n = precursors_n.to(model.decoder.device)
                memories_n = memories_n.to(model.decoder.device)
                mem_masks_n = mem_masks_n.to(model.decoder.device)
    
                (pred, return_tokens) = model.decoder(leaves, precursors_n, memories_n, mem_masks_n, partial=True)
    
                return_tokens = einops.repeat(return_tokens, "B L -> B L S", S=beam)
    
                pep_size = return_tokens.shape[1]
               
                tokens[:, :pep_size, :] = return_tokens
                tokens[:, pep_size, :] = torch.topk(pred[:, -1, :], beam, dim=1)[1]
    
                subpeplen = pred.shape[1]
    
                scores[:, :subpeplen, :, :] = einops.repeat(pred, "B L V -> B L V S", S=beam)
                scores[:, :subpeplen-1, :, :] = torch.nan
    
                # Make all tensors the right shape for decoding.
                precursors_n = einops.repeat(precursors_n, "B L -> (B S) L", S=beam)
                mem_masks_n = einops.repeat(mem_masks_n, "B L -> (B S) L", S=beam)
                memories_n = einops.repeat(memories_n, "B L V -> (B S) L V", S=beam)
    
                tokens = einops.rearrange(tokens, "B L S -> (B S) L")
                scores = einops.rearrange(scores, "B L V S -> (B S) L V")
    
                for step in range(pep_size, self._pep_len):
                    # Terminate beams exceeding the precursor m/z tolerance and track
                    # all finished beams (either terminated or stop token predicted).
                    (
                        finished_beams,
                        beam_fits_precursor,
                        discarded_beams,
                    ) = model.my_finish_beams(tokens, precursors_n, step, tail_mass)
    
                    # Cache peptide predictions from the finished beams (but not the
                    # discarded beams).
                    model.my_cache_finished_beams(
                        tokens,
                        scores,
                        step,
                        finished_beams & ~discarded_beams,
                        beam_fits_precursor,
                        pred_cache,
                    )
    
                    # Stop decoding when all current beams have been finished.
                    # Continue with beams that have not been finished and not discarded.
                    finished_beams |= discarded_beams
                    if finished_beams.all():
                        break
    
                    # Update the scores.
                    scores[~finished_beams, : step + 2, :], _ = model.decoder(
                        tokens[~finished_beams, : step + 1],
                        precursors_n[~finished_beams, :],
                        memories_n[~finished_beams, :, :],
                        mem_masks_n[~finished_beams, :],
                    )
    
                    scores[:, :subpeplen-1, :] = torch.nan
    
                    # Find the top-k beams with the highest scores and continue decoding
                    # those.
                    #tokens, scores = model.my_get_topk_beams(
                    tokens, scores = model._get_topk_beams(
                        tokens, scores, finished_beams|beam_fits_precursor, batch, step + 1
                    )
    
                # Return the peptide with the highest confidence score, within the
                # precursor m/z tolerance if possible.
                top_peptide_arr = list(model._get_top_peptide(pred_cache))
    
                for i,k in enumerate(leaf_keys):
                    if(top_peptide_arr[i]):
                        perfect_tree[k][delta] = top_peptide_arr[i][0][0]
                    else:
                        perfect_tree[k][delta] = 0.0
    
                gc.collect()
                torch.cuda.empty_cache()

        print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)
        return(perfect_tree)
        
    def parse_subtree(self, tree=None):
        #tree.show(filter = lambda x: x.data.ground_truth==True, idhidden=False, data_property='docking')
        children_nids = sorted([x.identifier for x in tree.children('root')])

        path_arr = []
        docking_arr = []
        ground_arr = []
        for c_nid in children_nids:
            path = []
            dockings = []
            ground_truth = tree[c_nid].data.ground_truth
            ground_arr.append(int(ground_truth))

            id_arr = tree.expand_tree(nid=c_nid, mode=2, sorting=True)
            for i in id_arr:
                g=tree[i].data.ground_truth
                if(g):
                    path_arr.append(i)
                d=tree[i].data.docking
                dockings.append(d)
            docking_arr.append(dockings)

        return(children_nids, path_arr, docking_arr, ground_arr)
   
    def parse_tree(self, tree=None, direction=None, abs_dock=False):
        #print('*'*50+'('+direction+')'+'*'*50)
        #tree.show(filter = lambda x: x.data.ground_truth==True, idhidden=False, data_property='mass')
        #tree.show(idhidden=False, data_property='mass')
        nodes_mass_dict=dict()
        for k,v in tree.nodes.items():
            nodes_mass_dict[k]=v.data.mass

        nodes_mass_list = list(nodes_mass_dict.values())
        nodes_nid_list = list(nodes_mass_dict.keys())

        terminal_mass = cp.array(nodes_mass_list, dtype=self._total_mass.dtype)
        probes, tokens = self.make_tensor_m(self._total_mass, terminal_mass, direction)

        docking = self.dock(probes)
        docking = cp.asnumpy(cp.squeeze(docking, axis=-1))

        for i,nid in enumerate(nodes_nid_list):
            if(abs_dock):
                tree[nid].data.docking=abs(docking[i])
            else:    
                tree[nid].data.docking=docking[i]

        #tree.show(filter = lambda x: x.data.ground_truth==True, idhidden=False, data_property='docking')
        #tree.show(idhidden=False, data_property='docking')
        #tree.show(idhidden=False, data_property='layer')

        #(children_nids, path_arr, docking_arr, ground_arr) = self.parse_subtree(tree)
        #return(children_nids, path_arr, docking_arr, ground_arr)

        return(True)

    def update_tree(self, tree, i, peptide_arr):
        new_tree = Tree()
        new_tag  = tree['root'].tag + peptide_arr[i]
        new_mass = tree['root'].data.mass + self._mass_dict[peptide_arr[i]]
        new_tree.create_node(tag=new_tag, identifier='root', data=MCTTS_Node(mass=new_mass, ground_truth=True))
        return(new_tree)

    def plant_forest(self):
        N_tree = Tree()
        C_tree = Tree()
        N_tree.create_node(tag='<', identifier='root', data=MCTTS_Node(mass=self._mass_dict['<'], ground_truth=True))
        C_tree.create_node(tag='>', identifier='root', data=MCTTS_Node(mass=self._mass_dict['>'], ground_truth=True))

        self._remaining_mass = float(self._total_mass) - N_tree['root'].data.mass - C_tree['root'].data.mass - self._proton
        
        N_forest = []
        C_forest = []
        #print('-'*100)
        #print(self._true_peptide)

        middle = math.ceil(len(self._peptide_arr)/2)+1
        docking_arr_average = dict()

        l=0
        for i in range(middle):
            #print('-'*70)
            #print(self._peptide_arr)

            N_tree = self.plant_tree(N_tree, i, self._peptide_arr)
            (children_nids, path_arr, docking_arr, ground_arr) = self.parse_tree(N_tree, 'left')
            N_forest.append([docking_arr, ground_arr])
            N_tree = self.update_tree(N_tree, i, self._peptide_arr)
        
            '''    
            for j in range(len(ground_arr)):
                label=ground_arr[j]
                if(label not in docking_arr_average):
                    docking_arr_average[label]=[0 for k in range(self._probe_layers)]
                for m in range(len(docking_arr[j])):
                    docking_arr_average[label][m] += abs(docking_arr[j][m])
                l+=1
            '''
    
            C_tree = self.plant_tree(C_tree, -1-i, self._peptide_arr)
            (children_nids, path_arr, docking_arr, ground_arr) = self.parse_tree(C_tree, 'right')
            C_forest.append([docking_arr, ground_arr])
            C_tree = self.update_tree(C_tree, -1-i, self._peptide_arr)

            '''
            for j in range(len(ground_arr)):
                label=ground_arr[j]
                if(label not in docking_arr_average):
                    docking_arr_average[label]=[0 for k in range(self._probe_layers)]
                for m in range(len(docking_arr[j])):
                    docking_arr_average[label][m] += abs(docking_arr[j][m])
                l+=1
            '''
        '''
        print('docking_arr_average')
        for k in docking_arr_average:
            print(k,end=':')
            arr=docking_arr_average[k]
            for s in arr:
                print(s/l,end=' ')
            print('\n')
        #pp.pprint(docking_arr_average)
        #print(l)
        '''
        return(N_forest, C_forest)

    def educate_children(self, node, direction):
        tree = Tree()
        tree.create_node(tag=node.residue, identifier='root', data=MCTTS_Node(mass=node.cumulative_mass, ground_truth=False))
        (tree, hierarchical_nodes) = self.plant_tree2(tree)
        #(children_nids, path_arr, docking_arr, ground_arr) = self.parse_tree(tree, direction, True)
        self.parse_tree(tree, direction, True)

        #tree.show(idhidden=False, data_property='docking')        
        for i in sorted(hierarchical_nodes.keys(), reverse=True):
            for nid in hierarchical_nodes[i]:
                children = tree.children(nid)
                children_dockings = [x.data.docking for x in children]
                min_docking = min(children_dockings)
                tree[nid].data.docking += min_docking
                if(i==1):
                    tree[nid].data.docking = tree[nid].data.docking/float(self._probe_layers)
        #tree.show(idhidden=False, data_property='docking')
        #print('-'*100)

        for child in node.children:
            nid='root_'+child.residue
            child_docking = self.activation(tree[nid].data.docking)
            #child_docking = tree[nid].data.docking
            #child.docking = child_docking

            child.reward = child_docking
        return(True)

    #def BACKUP(self, node, reward, docking):
    def BACKUP(self, node, reward):
        while(node.parent != None):
            node.visits += 1
            node.reward += reward
            #node.docking += docking
            node=node.parent
        return(True)

    def check_diff(self, diff, title='diff'):
        print(title,end=', shape:')
        print(diff.shape, end=', size:')
        print(len(cp.ravel(diff)), end=', type:')
        print(type(diff))
        print(diff)
        print('-'*10)

    def search_closest(self, query):
        if(query >= self._sorted_peptides_mass_arr[-1]):
            return(query - self._sorted_peptides_mass_arr[-1])
        elif(query <= self._sorted_peptides_mass_arr[0]):
            return(self._sorted_peptides_mass_arr[0] - query)
        
        pos = bisect.bisect_left(self._sorted_peptides_mass_arr, query)

        before = self._sorted_peptides_mass_arr[pos-1]
        after = self._sorted_peptides_mass_arr[pos]
        diff=min(query - before, after - query)

        return(diff)

    def DEFAULTPOLICY_bisect(self, tail_mass):
        diff=self.search_closest(tail_mass)
        return(diff)

    def DEFAULTPOLICY(self, current_tail_mass):
        reward = abs(current_tail_mass)
        values = cp.asarray(self._tokens_values)
        #self.check_diff(values, 'values')

        diff = cp.asarray(current_tail_mass)
        #self.check_diff(diff, 'diff initial')

        while(True):
            diff = cp.extract(diff > 0.0, diff)
            #self.check_diff(diff, 'diff extract')

            diff_total_size = len(cp.ravel(diff))
            if( diff_total_size <=0 ):
                #print('\n'.join(['(<=0)'*50 for i in range(1)]))
                break
            elif( diff_total_size > self._size_upper_limit ):
                diff = cp.random.choice(diff, size=self._size_upper_limit, replace=False)
                #self.check_diff(diff, 'diff random choice')

            diff = cp.reshape(diff, [-1,1])
            #self.check_diff(diff, 'After ceiling and reshape')

            diff = diff - values
            #self.check_diff(diff, 'diff - values')

            diff = cp.ravel(diff)
            #self.check_diff(diff, 'diff ravel')

            diff_abs = cp.absolute(diff)
            #self.check_diff(diff_abs, 'diff_abs')

            diff_min = cp.amin(diff_abs).item()
            #print('diff_min', end=':')
            #print(diff_min, type(diff_min))
            
            if(diff_min<reward):
                reward=diff_min
        return(reward)

    #current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
    def BESTCHILD(self, node=None, scalar=0.0, most=False):
        if(node==None):
            sys.exit('Node must not be None')

        #bestscore=0.0
        bestscore=float('-inf')
        bestchild=None

        most_visited = 0
        most_visited_child = None

        for c in node.children:

            exploit = c.reward/c.visits
            explore = math.sqrt(math.log(node.visits)/float(c.visits))
            score = exploit + scalar * explore

            if(score >= bestscore):
                bestchild=c
                bestscore=score
            if(c.visits > most_visited):
                most_visited_child = c
                most_visited = c.visits
                
        if(bestchild == None):
            logger.warn("NO child found, check please!")
        if(most_visited_child == None):
            logger.warn("NO most_visited_child found, check please!")

        if(most):
            bestchild=most_visited_child

        return(bestchild)
    
    def EXPAND(self, node, direction=None):
        tried = [c.residue for c in node.children]
        if(tried):
            if(len(tried) != len(self._tokens_keys)):
                sys.exit("Node's children must be none or full!")
        else:
            untried = list(set(self._tokens_keys).difference(set(tried)))
            node.children = [myNode(residue, self._mass_dict[residue], node.layer-1, parent=node) for residue in untried]
            self.educate_children(node, direction)
        return(True)
    
    def EXPAND_Transformer(self, perfect_tree, node, mode):
        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' started '+ '+'*100)
        tried = [c.residue for c in node.children]
        if(tried):
            if(len(tried) != len(self._tokens_keys)):
                sys.exit("Node's children must be none or full!")
        else:
            untried = list(set(self._tokens_keys).difference(set(tried)))
            for residue in untried:
                child=myNode(residue, self._mass_dict[residue], node.layer-1, parent=node)

                key_j = node.sequence.replace(',','_') +'_'+residue
                if(key_j not in perfect_tree):
                    sys.exit('Error: ' + key_j + 'is not in perfect_tree')
                child.reward = perfect_tree[key_j][mode]
                child.delta[-1] = perfect_tree[key_j][-1]
                child.delta[-2] = perfect_tree[key_j][-2]
                child.delta[-3] = perfect_tree[key_j][-3]
                child.delta[-4] = perfect_tree[key_j][-4]

                node.children.append(child)

        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)
        return(True)

    def TREEPOLICY_single(self, root, tail_mass, direction=None):
        if(direction==None):
            sys.exit('direction must NOT be none')
        leaf = root

        while(not leaf.is_terminal()):
            if(len(leaf.children) < len(self._tokens_keys)):
                self.EXPAND(leaf, direction)
            leaf = self.BESTCHILD(node=leaf, scalar=self._SCALAR, most=False)
            #print('leaf.residue',end=':')
            #print(leaf.residue)
            tail_mass -= leaf.mass
        return(leaf, tail_mass)

    def TREEPOLICY_Transformer(self, perfect_tree, root, tail_mass, mode):
        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' started '+ '+'*100)
        leaf = root

        while(not leaf.is_terminal()):
            if(len(leaf.children) < len(self._tokens_keys)):
                self.EXPAND_Transformer(perfect_tree, leaf, mode)
            leaf = self.BESTCHILD(node=leaf, scalar=self._SCALAR_Transformer, most=False)
            tail_mass -= leaf.mass

        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)
        return(leaf, tail_mass)

    def give_birth_to_single(self, root=None, direction=None, other_root=None):
        if(root==None or direction==None or other_root==None):
            sys.exit('Root and direction must NOT be None!')

        if(direction=='left'):
            other_terminal_mass = self._mass_dict['>']
        elif(direction=='right'):
            other_terminal_mass = self._mass_dict['<']
        else:
            sys.exit('Direction must be left or right!')

        i=0
        while(i < self._budget):
            gap_mass = float(self._total_mass) - root.mass - other_root.mass #gap mass

            #start=time.time()
            (leaf, gap_mass) = self.TREEPOLICY_single(root, gap_mass, direction)
            #end=time.time()

            reward = leaf.reward
            if(gap_mass<250):
                gap_docking = self.DEFAULTPOLICY_bisect(gap_mass)
                reward = self.activation(gap_docking)


            start=time.time()
            self.BACKUP(leaf, reward)
            end=time.time()
            i+=1
        #bestchild = self.BESTCHILD(node=root, scalar=0.0, most=True)
        bestchild = self.BESTCHILD(node=root, scalar=0.0, most=False)
        self._remaining_mass -= bestchild.mass
        return(bestchild)

    #def UCTSEARCH(self, spectrum, precursor, peptide):
    def UCTSEARCH(self, arr):
        [spectrum, precursor, peptide]=arr
        p=precursor.tolist()[0]
        self._total_mass = np.double(p[0])
        charge_num = int(p[1])
        self.mz_value = np.double([2])

        self._charge = self.readin_charge(charge_num)

        '''
        self._utils.parse_var(precursor)
        self._utils.parse_var(self._total_mass, 'self._total_mass')
        self._utils.parse_var(self._charge, 'self._charge')
        self._utils.parse_var(self.mz_value, 'self.mz_value')
        '''
        self.branches_arr = self.make_branches(self._depth_Transformer)
        '''
        self._utils.parse_var(spectrum, 'A')
        self._utils.parse_var(precursor)
        self._utils.parse_var(peptide)
        '''

        self._spectrum = spectrum
        self._mz = self.readin_spectrum(spectrum)

        #self._utils.parse_var(self._mz,'self._mz')
        #sys.exit(0)

        N_root = myNode(residue='<', mass=self._mass_dict['<'], layer= self._depth)
        C_root = myNode(residue='>', mass=self._mass_dict['>'], layer= self._depth)
        self._remaining_mass = float(self._total_mass) - N_root.mass - C_root.mass - self._proton

        while(self._remaining_mass > 0.0):
            N_bestchild = self.give_birth_to_single(N_root, 'left', C_root)
            N_root = myNode(residue=N_root.residue+','+N_bestchild.residue, mass=N_root.mass+N_bestchild.mass, layer= self._depth)
            if(self._remaining_mass <=0.0):
                break
            C_bestchild = self.give_birth_to_single(C_root, 'right', N_root)
            C_root = myNode(residue=C_bestchild.residue+','+C_root.residue, mass=C_root.mass+C_bestchild.mass, layer= self._depth)

        pred_peptide = N_root.residue + ',' + C_root.residue
        pred_mass = N_root.mass + C_root.mass
        true_mass = self._total_mass

        mass_error = abs(pred_mass - true_mass)

        #self._pred_peptide = pred_peptide.replace('<,','').replace(',>','')
        #result = [self._true_peptide, self._pred_peptide, self._true_peptide==self._pred_peptide, true_mass, pred_mass, mass_error]

        pred_peptide = pred_peptide.replace('<,','').replace(',>','')
        true_peptide = ','.join(peptide[0])

        result = [true_peptide, pred_peptide, true_peptide==pred_peptide, true_mass, pred_mass, mass_error]

        return(result)

    '''
    def test_Transformer_with_beam(self, spectrum, precursor):
        peptides_pred = []
        for spectrum_preds in self.Transformer_N.forward(spectrum, precursor):
            if(not spectrum_preds):
                peptides_pred.append('')
                continue
            for _, _, pred in spectrum_preds:
                if(not pred):
                    pred=''
                peptides_pred.append(pred)

        for spectrum_preds in self.Transformer_C.forward(spectrum, precursor):
            if(not spectrum_preds):
                peptides_pred.append('')
                continue
            for _, _, pred in spectrum_preds:
                if(not pred):
                    pred=''
                peptides_pred.append(pred)

        return(peptides_pred)
    '''

    def give_birth_to_Transformer(self, precursor, total_mass, N_root, N_memory, N_mem_mask, C_root, C_memory, C_mem_mask, mode, delta):
        print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' started '+ '+'*100)
        
        N_tail_mass = total_mass - self._mass_dict['<'] - C_root.mass
        C_tail_mass = total_mass - self._mass_dict['>'] - N_root.mass

        start=time.time()
        with torch.no_grad():
            perfect_N_tree = self.plant_perfect_tree(precursor, N_root, N_tail_mass, self.Transformer_N, N_memory, N_mem_mask, mode, delta)
            perfect_C_tree = self.plant_perfect_tree(precursor, C_root, C_tail_mass, self.Transformer_C, C_memory, C_mem_mask, mode, delta)
        end=time.time()
        print('time_consumed in self.plant_perfect_tree with mode '+str(mode),end=':')
        print(end-start)

        #if(delta==-4):
        #    self._utils.parse_var(perfect_N_tree)

        start=time.time()
        i=0
        while(i < self._budget):
            #print('('+str(i)+')'+'-'*100)
            gap_mass = total_mass - N_root.mass - C_root.mass

            #start=time.time()
            (N_leaf, N_tail_mass) = self.TREEPOLICY_Transformer(perfect_N_tree, N_root, gap_mass, mode)
            (C_leaf, C_tail_mass) = self.TREEPOLICY_Transformer(perfect_C_tree, C_root, gap_mass, mode)
            #end=time.time()

            #print('time_consumed in self.TREEPOLICY_Transformer',end=':')
            #print(end-start)

            #if(self.bisect_search_delta):
            if(self._ceiling and delta==-2):
                start=time.time()
                N_leaf.delta[delta] = self.activation(self.DEFAULTPOLICY_bisect(N_tail_mass))
                C_leaf.delta[delta] = self.activation(self.DEFAULTPOLICY_bisect(C_tail_mass))
                end=time.time()
                print('time_consumed in self.DEFAULTPOLICY_bisect',end=':')
                print(end-start)
            
            N_delta = N_leaf.delta[delta]
            C_delta = C_leaf.delta[delta]

            '''
            if(delta==-2 and N_tail_mass>=250):
                N_delta = N_leaf.reward
            if(delta==-2 and C_tail_mass>=250):
                C_delta = C_leaf.reward
            '''

            #start=time.time()
            self.BACKUP(N_leaf, N_delta)
            self.BACKUP(C_leaf, C_delta)
            #end=time.time()

            i+=1
        end=time.time()
        print('time_consumed in self.TREEPOLICY_Transformer and self.BACKUP in '+str(i)+ ' iterations',end=':')
        print(end-start)

        N_bestchild = self.BESTCHILD(node=N_root, scalar=0.0, most=False)
        C_bestchild = self.BESTCHILD(node=C_root, scalar=0.0, most=False)
        print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)
        return(N_bestchild, C_bestchild)

    #def UCTSEARCH_Transformer(self, N_memory, N_mem_mask, C_memory, C_mem_mask, precursor, peptide, mode=0, delta=-1):
    def UCTSEARCH_Transformer(self, arr):
        [N_memory, N_mem_mask, C_memory, C_mem_mask, precursor, peptide, mode, delta] = arr
        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' started '+ '+'*100)
        #mode=0 Transformer mode
        #mode=1 Spectrum Probes mode

        '''
        print('delta',end=':')
        print(delta)
        print('self._depth_Transformer',end=':')
        print(self._depth_Transformer)
        '''

        self.branches_arr = self.make_branches(self._depth_Transformer)

        #precursor
        p = precursor[0].tolist()
        total_mass = p[0]


        #peptide
        true_peptide = self.get_peptide_true(peptide[0])
        #self._utils.parse_var(self._mass_dict)
        #self._utils.parse_var(self._meta.tokens)

        N_root = myNode(residue='<', mass=self._mass_dict['<'], layer = self._depth_Transformer)
        C_root = myNode(residue='>', mass=self._mass_dict['>'], layer = self._depth_Transformer)

        remaining_mass = total_mass - N_root.mass - C_root.mass - self._proton

        #self._utils.parse_var(remaining_mass)

        while(remaining_mass > 0.0):
            N_bestchild, C_bestchild = self.give_birth_to_Transformer(
                precursor,
                total_mass,
                N_root,
                N_memory,
                N_mem_mask,
                C_root,
                C_memory,
                C_mem_mask,
                mode,
                delta
            )

            N_root = myNode(residue=N_root.residue+','+N_bestchild.residue, mass=N_root.mass+N_bestchild.mass, layer= self._depth_Transformer)

            remaining_mass -= N_bestchild.mass

            if(remaining_mass<=0.0):
                break
            C_root = myNode(residue=C_root.residue+','+C_bestchild.residue, mass=C_root.mass+C_bestchild.mass, layer= self._depth_Transformer)
            remaining_mass -= C_bestchild.mass
       
        pred_peptide = N_root.residue.split(',') + C_root.residue.split(',')[::-1]
        pred_peptide = ','.join(pred_peptide)
        pred_peptide = pred_peptide.replace('<,','').replace(',>','')
        
        pred_mass = N_root.mass + C_root.mass
        true_mass = total_mass
        mass_error = abs(pred_mass - true_mass)
        
        result = [true_peptide, pred_peptide, true_peptide==pred_peptide, true_mass, pred_mass, mass_error]
        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)

        #del N_memory, N_mem_mask, C_memory, C_mem_mask, precursor, peptide, mode, delta
        #torch.cuda.empty_cache()
        return(result)

    def UCTSEARCH_final(self, spectrum, precursor, peptide):
        pass
