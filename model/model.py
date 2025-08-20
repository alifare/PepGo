#The development began around 2019-02-21
import os
import re
import sys

import time
import pathlib
import pandas as pd
import warnings
import numpy as np
np.set_printoptions(suppress=True)

from tqdm import tqdm
from collections import OrderedDict
from typing import Union

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Manager, Pool

import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint

from .Transformer import Transformer
from .MCTTS import Monte_Carlo_Double_Root_Tree
from .utils import UTILS

class SpecDataSet(torch.utils.data.Dataset):
    def __init__(self, spec_file, reverse=False):
        super().__init__()
        self.spec_file = spec_file
        self.reverse = reverse

        self._max_peak_num = 300
        spec_offset_dict, lines_num = self.read_pickle(spec_file)
        lines=self.read_lines(spec_file, spec_offset_dict)

        self.spec_dict = []
        for i in lines:
            pep = self._parse_line(i)
            sample = self.pep_to_sample(pep)
            self.spec_dict.append(sample)

    def __getitem__(self, idx):
        spec=self.spec_dict[idx]
        return(spec)

    def __len__(self):
        size=len(self.spec_dict)
        return(size)

    def _parse_line(self, line):
        line=line.strip()
        m=re.search('^#',line)
        if(m or line==''):
            return(False)

        pep=dict()
        arr=line.split('\t')
        pep['peptide']=arr.pop(0)
        pep['Naa']=len(pep['peptide'])
        pep['charge']=int(arr.pop(0))
        pep['mw']=float(arr.pop(0))
        pep['Mods_num']=int(arr.pop(0))
        pep['Mods']=arr.pop(0)
        pep['iRT']=arr.pop(0)
        Collision=arr.pop(0)
        spec_id=arr.pop(0)
        pep['spec_id']=spec_id
        pep['Num_peaks']=int(arr.pop(0))
        pep['ions']=arr.pop(0).split(',')
        return(pep)

    def pep_to_sample(self, pep):
        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' started '+ '+'*100)
        peptide = pep['peptide']
        charge = pep['charge']
        mw = pep['mw']
        Mods_num = pep['Mods_num']
        Mods = pep['Mods']
        Num_peaks = pep['Num_peaks']

        peptide = re.sub('[IL]', 'X', peptide)

        ions=pep['ions']
        spec_id=pep['spec_id']

        peak_num = len(ions)
        if(self._max_peak_num < peak_num):
            self._max_peak_num = peak_num

        x = [ [float(j) for j in i.split(':')] for i in ions ]

        y = self.peptide_to_seqarr(peptide, Mods_num, Mods)

        s = [float(mw)]

        c = [int(charge)]

        sample = [x,y,s,c]
        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)
        return(sample)

    def peptide_to_seqarr(self, peptide, Mods_num, Mods):
        side_chain = [''] * len(peptide)
        if(Mods_num):
            for mod in Mods.split('/'):
                (pos, residue, ptm) = mod.split(',')
                pos = int(pos)
                side_chain[pos] = ptm

        seqarr = []
        for i,r in enumerate(peptide):
            ptm=side_chain[i]
            if(ptm):
                r = r+'+'+ptm
            seqarr.append(r)
            #residue_matrix[i, 0]=self.convert_residue_to_onehot(r)

        if(self.reverse):
            seqarr = seqarr[::-1]

        return(seqarr)

    def peptide_to_seqarr_startend(self, peptide, Mods_num, Mods):
        peptide = self._start + peptide + self._end

        side_chain = [''] * len(peptide)
        if(Mods_num):
            for mod in Mods.split('/'):
                (pos, residue, ptm) = mod.split(',')
                pos = int(pos)+1 #Very important, because the start token has been added to the head of the peptide!
                side_chain[pos] = ptm

        seqarr = []
        for i,r in enumerate(peptide):
            ptm=side_chain[i]
            if(ptm):
                r = r+'+'+ptm
            seqarr.append(r)
            #residue_matrix[i, 0]=self.convert_residue_to_onehot(r)

        if(self.reverse):
            seqarr = seqarr[::-1]

        return(seqarr)

    @classmethod
    def _iter_count(sel, spec_file):
        from itertools import (takewhile, repeat)
        buffer = 1024 * 1024
        with open(spec_file) as f:
            buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
            return(sum(buf.count('\n') for buf in buf_gen))

    @classmethod
    def _make_data_offset_dict(self, spec_file):
        total_lines_num = self._iter_count(spec_file)
        offset_dict = OrderedDict()
        with open(spec_file, 'rb') as f:
            offset_dict[0]=f.tell()
            for line_num, _ in tqdm(enumerate(f), total=total_lines_num):
                offset_dict[line_num+1]=f.tell()
        offset_dict.popitem() # remove last key
        return(offset_dict)

    @classmethod
    def _save_pickle(self, offset_dict, data_offset_dict_file):
        f_out = open(data_offset_dict_file, 'w')
        for line in offset_dict:
            offset = offset_dict[line]
            f_out.write(str(line) +'\t'+ str(offset)+'\n')
        f_out.close()
        return(True)

    @classmethod
    def index_file(self, spec_file):
        offset_dict = self._make_data_offset_dict(spec_file)
        self._save_pickle(offset_dict, spec_file+'.offset')

    @classmethod
    def read_pickle(self, spec_file):
        offset_file = spec_file + '.offset'
        path = pathlib.Path(offset_file)
        if(not (path.exists() and path.is_file())):
            self.index_file(spec_file)

        total_lines_num = self._iter_count(spec_file)
        lines_num = total_lines_num - 1
        offset_dict = OrderedDict()
        f_in = open(offset_file, 'r')
        for line in f_in:
            (i, offset)=line.strip().split('\t')
            offset_dict[int(i)]=int(offset)
        f_in.close()
        return(offset_dict, lines_num)

    def read_lines(self, spec_file, spec_offset_dict):
        with open(spec_file, 'r') as f:
            arr=[]
            for k in spec_offset_dict:
                offset = spec_offset_dict[k]
                f.seek(offset)
                line = f.readline().strip()
                if(re.match('^#', line)):
                    continue
                arr.append(line)
            return arr

    def load_spec(self, batch_size, spec_file):
        print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' started '+ '+'*100)
        self.batch_size = batch_size
        offset_dict, lines_num = self.read_pickle(spec_file)
        lines=self.random_read_lines(spec_file, offset_dict, [3,10])
        print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)


class MODEL:
    def __init__(self, meta, configs):
        super().__init__()
        self._meta = meta
        self._proton = self._meta.proton
        self._configs = configs
        self._utils = UTILS()

        '''
        self._utils.parse_var(meta.tokens)
        self._utils.parse_var(meta.special_tokens)
        self._utils.parse_var(meta.residues)
        self._utils.parse_var(meta.mass_dict)
        self._utils.parse_var(self._configs)
        '''

        #self._Transformer = Transformer(meta=self._meta, configs=self._configs)
        #self._utils.parse_var(self._Transformer)
        #self._mctts = Monte_Carlo_Double_Root_Tree(meta=self._meta, configs=self._configs)

        # Initialized later:
        self.tmp_dir = None
        self.trainer = None
        self.model = None
        self.loaders = None
        self.writer = None


    def spec_collate(self, item):
        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' started '+ '+'*100)
        spectra = []
        peptides = []
        total_mass = []
        charge = []
        precursors = []

        for i in item:
            #spectra.append(torch.tensor(i[0]))
            s=torch.tensor(i[0])

            int_array = torch.sqrt(s[:,1])
            int_array /= torch.linalg.norm(int_array)
            s[:,1] = int_array

            spectra.append(s)

            peptides.append(i[1])
            total_mass.append(i[2])
            charge.append(i[3])

            #calc_mass = (calc_mass / charge) + self._proton
            mz = (i[2][0] / i[3][0]) + self._proton

            precursors.append([i[2][0], i[3][0], mz])

        spectra = torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True)
        precursors = torch.tensor(precursors)

        batch = [spectra, precursors, peptides]

        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)
        return(batch)

    def train(self, train_spec=None, valid_spec=None):
        print(train_spec)
        print(valid_spec)

        #Training self.Transformer_N
        train_spec_set = SpecDataSet(train_spec, False)
        train_spec_set_loader = torch.utils.data.DataLoader(
            train_spec_set,
            batch_size=self._configs['Model']['Trainer']['train_batch_size'],
            num_workers=self._configs['Model']['Trainer']['num_workers'],
            collate_fn=self.spec_collate,
            shuffle=True,
        )

        valid_spec_set = SpecDataSet(valid_spec, False)
        valid_spec_set_loader = torch.utils.data.DataLoader(
            valid_spec_set,
            batch_size=self._configs['Model']['Trainer']['valid_batch_size'],
            num_workers=self._configs['Model']['Trainer']['num_workers'],
            collate_fn=self.spec_collate
        )

        self.trainer_N.fit(self.Transformer_N, train_dataloaders=train_spec_set_loader, val_dataloaders=valid_spec_set_loader)
        del train_spec_set, valid_spec_set

        #Training self.Transformer_C
        train_spec_set = SpecDataSet(train_spec, True)
        train_spec_set_loader = torch.utils.data.DataLoader(
            train_spec_set,
            batch_size=self._configs['Model']['Trainer']['train_batch_size'],
            num_workers=self._configs['Model']['Trainer']['num_workers'],
            collate_fn=self.spec_collate,
            shuffle=True,
        )

        valid_spec_set = SpecDataSet(valid_spec, True)
        valid_spec_set_loader = torch.utils.data.DataLoader(
            valid_spec_set,
            batch_size=self._configs['Model']['Trainer']['valid_batch_size'],
            num_workers=self._configs['Model']['Trainer']['num_workers'],
            collate_fn=self.spec_collate
        )

        self.trainer_C.fit(self.Transformer_C, train_dataloaders=train_spec_set_loader, val_dataloaders=valid_spec_set_loader)
        del train_spec_set, valid_spec_set

    def predict(self, spec_file=None):
        mp.set_start_method('spawn', force=True)

        out_file = os.path.basename(spec_file) \
                +'.depth'+str(self._configs['MCTTS']['Tree']['depth']) \
                +'.probe_layers'+str(self._configs['MCTTS']['Tree']['probe_layers']) \
                +'.depth_Transformer'+str(self._configs['MCTTS']['Tree']['depth_Transformer']) \
                +'.depth_Transformer_beam'+str(self._configs['MCTTS']['Tree']['depth_Transformer_beam']) \
                +'.ceiling'+str(self._configs['MCTTS']['Delta']['ceiling']) \
                +'.budget'+str(self._configs['MCTTS']['Tree']['budget']) \
                +'.T_beam_search'+str(int(self._configs['MCTTS']['Delta']['mode']['transformer_beam_search'])) \
                +'.beam'+str(self._configs['Model']['Transformer']['n_beams']) \
                +'.gap_mass.result.txt'
        f_out = open(out_file,'w')
        f_out.write('#true_peptide\tpred_peptide\tmatched\ttrue_mass\tpred_mass\tmass_error\t')
        f_out.write('probe\tT_bisect\tT_beam\n')

        monte = Monte_Carlo_Double_Root_Tree(meta=self._meta, configs=self._configs,
                Transformer_N=self.Transformer_N, Transformer_C=self.Transformer_C)

        spec_set = SpecDataSet(spec_file, False)
        spec_set_loader = torch.utils.data.DataLoader(
            spec_set,
            batch_size=self._configs['Model']['Trainer']['test_batch_size'],
            num_workers=self._configs['Model']['Trainer']['num_workers'],
            collate_fn=self.spec_collate
        )

        for item in spec_set_loader:
            start=time.time()
            spectra, precursors, peptides = item
            test_batch_size = spectra.shape[0]

            spectra = spectra.to(self.Transformer_N.encoder.device)
            N_memories, N_mem_masks = self.Transformer_N.encoder(spectra)
            C_memories, C_mem_masks = self.Transformer_C.encoder(spectra)

            lines_probe = []
            lines_bisect = []
            lines_beam = []

            final_results = []
            for i in range(test_batch_size):
                spectrum = spectra[i:i+1]
                precursor = precursors[i:i+1]
                peptide = peptides[i:i+1]

                N_memory = N_memories[i:i+1]
                N_mem_mask = N_mem_masks[i:i+1]
                C_memory = C_memories[i:i+1]
                C_mem_mask = C_mem_masks[i:i+1]

                lines_probe.append([spectrum, precursor, peptide])
                lines_bisect.append([N_memory.detach(), N_mem_mask.detach(), C_memory.detach(), C_mem_mask.detach(), precursor.detach(), peptide, 0, -2])
                lines_beam.append([N_memory.detach(), N_mem_mask.detach(), C_memory.detach(), C_mem_mask.detach(), precursor.detach(), peptide, 0, -4])

            result_probe = [['-','-', False, 0.0, 0.0, 100000.0] for i in range(test_batch_size)]
            result_T_bisect = [['-','-', False, 0.0, 0.0, 100000.0] for i in range(test_batch_size)]
            result_T_beam = [['-','-', False, 0.0, 0.0, 100000.0] for i in range(test_batch_size)]

            if(self._configs['MCTTS']['Delta']['mode']['probe_bisect_search']):
                with Pool(processes = test_batch_size) as pool:
                    result_probe = pool.map(monte.UCTSEARCH, lines_probe)

            if(self._configs['MCTTS']['Delta']['mode']['transformer_bisect_search']):
                with Pool(processes = test_batch_size) as pool:
                    result_T_bisect = pool.map(monte.UCTSEARCH_Transformer, lines_bisect)

            if(self._configs['MCTTS']['Delta']['mode']['transformer_beam_search']):
                tmp = monte._depth_Transformer
                monte._depth_Transformer = self._configs['MCTTS']['Tree']['depth_Transformer_beam']
                with Pool(processes = test_batch_size) as pool:
                    result_T_beam = pool.map(monte.UCTSEARCH_Transformer, lines_beam)
                monte._depth_Transformer = tmp

            for i in range(test_batch_size):
                precursor = precursors[i:i+1]
                peptide = peptides[i:i+1]

                true_peptide = ','.join(peptide[0])
                pred_peptide = result_T_bisect[i][1]
                matched = str(result_T_bisect[i][2])
                true_mass = str(precursor[0][0].item())
                pred_mass = str(result_T_bisect[i][4])
                mass_error = str(result_T_bisect[i][5])

                final_results = [true_peptide, pred_peptide, matched, true_mass, pred_mass, mass_error]

                results = [result_probe[i], result_T_bisect[i], result_T_beam[i]]
                for i,k in enumerate(results):
                    pred_peptide = k[1]
                    matched = str(k[2])
                    pred_mass = str(k[4])
                    mass_error = str(k[5])
                    r = ':'.join([pred_peptide, matched, pred_mass, mass_error])
                    final_results.append(r)

                f_out.write('\t'.join(final_results)+'\n')

            end=time.time()
            print('time_consumed in one prediction batch',end=':')
            print(end-start)
        f_out.close()

        torch.cuda.empty_cache()
        return(True)

    def initialize_trainer(self, train: bool) -> None:
        """Initialize the lightning Trainer.

        Parameters
        ----------
        train : bool
            Determines whether to set the trainer up for model training
            or evaluation / inference.
        """
        trainer_cfg_N = dict(
            accelerator=self._configs['Model']['Trainer']['accelerator'],
            devices=1,
            enable_checkpointing=False,
        )

        trainer_cfg_C = dict(
            accelerator=self._configs['Model']['Trainer']['accelerator'],
            devices=1,
            enable_checkpointing=False,
        )

        if train:
            #print(self._configs['Train']['devices'])
            if self._configs['Model']['Trainer']['devices'] is None:
                devices = "auto"
            else:
                self._configs['Model']['Trainer']['devices']

            additional_cfg_N = dict(
                devices = devices,
                callbacks = self.callbacks_N,
                enable_checkpointing = self._configs['Model']['Trainer']['save_top_k'] is not None,
                max_epochs = self._configs['Model']['Trainer']['epoch'],
                num_sanity_val_steps=self._configs['Model']['Trainer']['num_sanity_val_steps'],
                strategy=self._get_strategy(),
                #val_check_interval=self.config.val_check_interval,
                check_val_every_n_epoch=1,
                #check_val_every_n_epoch=None,
            )

            additional_cfg_C = dict(
                devices = devices,
                callbacks = self.callbacks_C,
                enable_checkpointing = self._configs['Model']['Trainer']['save_top_k'] is not None,
                max_epochs = self._configs['Model']['Trainer']['epoch'],
                num_sanity_val_steps=self._configs['Model']['Trainer']['num_sanity_val_steps'],
                strategy=self._get_strategy(),
                #val_check_interval=self.config.val_check_interval,
                check_val_every_n_epoch=1,
                #check_val_every_n_epoch=None,
            )

            trainer_cfg_N.update(additional_cfg_N)
            trainer_cfg_C.update(additional_cfg_C)

        self.trainer_N = pl.Trainer(**trainer_cfg_N)
        self.trainer_C = pl.Trainer(**trainer_cfg_C)

    def initialize_model(self, mode=None, models_dir=None) -> None:
        models_dir = os.path.normpath(models_dir)

        models_dir_N = os.path.join(models_dir, 'ckpt_N')
        models_dir_C = os.path.join(models_dir, 'ckpt_C')

        model_filename_N = os.path.join(models_dir_N, 'last.ckpt')
        model_filename_C = os.path.join(models_dir_C, 'last.ckpt')

        residues = self._meta.tokens
        model_params = dict(
            dim_model = self._configs["Model"]["Transformer"]['dim_model'],
            n_head = self._configs["Model"]["Transformer"]['n_head'],
            dim_feedforward = self._configs["Model"]["Transformer"]['dim_feedforward'],
            n_layers = self._configs["Model"]["Transformer"]['n_layers'],
            dropout = self._configs["Model"]["Transformer"]['dropout'],
            dim_intensity = self._configs["Model"]["Transformer"]['dim_intensity'],
            max_length = self._configs["Model"]["Transformer"]['max_length'],
            residues = residues,
            max_charge = self._configs["Model"]["Transformer"]['max_charge'],
            precursor_mass_tol = self._configs["Model"]["Transformer"]['precursor_mass_tol'],

            isotope_error_range = self._configs['Model']['Transformer']['isotope_error_range'],

            min_peptide_len = self._configs["Model"]["Transformer"]['min_peptide_len'],
            n_beams = self._configs["Model"]["Transformer"]['n_beams'],
            top_match = self._configs["Model"]["Transformer"]['top_match'],
            n_log = self._configs["Model"]["Transformer"]['n_log'],

            tb_summarywriter =self._configs['Model']['Transformer']['tb_summarywriter'],

            train_label_smoothing=self._configs['Model']['Transformer']['train_label_smoothing'],
            warmup_iters=self._configs['Model']['Transformer']['warmup_iters'],
            max_iters=self._configs['Model']['Transformer']['max_iters'],
            lr=self._configs['Model']['Trainer']['learning_rate'],
            weight_decay=self._configs['Model']['Trainer']['weight_decay'],
            out_writer=self.writer,
            calculate_precision=self._configs['Model']['Transformer']['calculate_precision'],
            meta = self._meta
        )

        # Reconfigurable non-architecture related parameters for a loaded model
        loaded_model_params = dict(
            max_length = self._configs["Model"]["Transformer"]['max_length'],
            precursor_mass_tol=self._configs['Model']['Transformer']['precursor_mass_tol'],
            isotope_error_range=self._configs['Model']['Transformer']['isotope_error_range'],
            n_beams=self._configs['Model']['Transformer']['n_beams'],
            min_peptide_len=self._configs['Model']['Transformer']['min_peptide_len'],
            top_match=self._configs['Model']['Transformer']['top_match'],
            n_log=self._configs['Model']['Transformer']['n_log'],
            tb_summarywriter=self._configs['Model']['Transformer']['tb_summarywriter'],
            train_label_smoothing=self._configs['Model']['Transformer']['train_label_smoothing'],
            warmup_iters=self._configs['Model']['Transformer']['warmup_iters'],
            max_iters=self._configs['Model']['Transformer']['max_iters'],
            lr=self._configs['Model']['Trainer']['learning_rate'],
            weight_decay=self._configs['Model']['Trainer']['weight_decay'],
            out_writer=self.writer,
            calculate_precision=self._configs['Model']['Transformer']['calculate_precision'],
            meta = self._meta
        )

        if(mode=='train'):
            if(not os.path.exists(models_dir_N)):
                os.makedirs(models_dir_N)
            if(not os.path.exists(models_dir_C)):
                os.makedirs(models_dir_C)

            # Configure checkpoints.
            if self._configs['Model']['Trainer']['save_top_k'] is not None:
                self.callbacks_N = [
                    ModelCheckpoint(
                        dirpath=models_dir_N,
                        monitor="valid_CELoss",
                        mode="min",
                        save_top_k=self._configs['Model']['Trainer']['save_top_k'],
                        save_last='link', #Added by ChangYuqi
                        enable_version_counter = False, #Added by ChangYuqi
                    )
                ]

                self.callbacks_C = [
                    ModelCheckpoint(
                        dirpath=models_dir_C,
                        monitor="valid_CELoss",
                        mode="min",
                        save_top_k=self._configs['Model']['Trainer']['save_top_k'],
                        save_last='link', #Added by ChangYuqi
                        enable_version_counter = False, #Added by ChangYuqi
                    )
                ]
            else:
                self.callbacks_N = None
                self.callbacks_C = None

            self.initialize_trainer(train=True)
            self.Transformer_N = Transformer(**model_params)
            self.Transformer_C = Transformer(**model_params)
        elif(mode=='predict'):
            if(not os.path.exists(model_filename_N) or not os.path.exists(model_filename_C)):
                sys.exit('Please check the directory of Transormer models!')

            self.initialize_trainer(train=False)

            device = torch.empty(1).device  # Use the default device.
            #device = 'cuda'
            device = None

            self._utils.parse_var(device)
            #self._utils.parse_var(loaded_model_params)

            try:
                self.Transformer_N = Transformer.load_from_checkpoint(
                    model_filename_N, map_location=device, **loaded_model_params
                )

                self.Transformer_C = Transformer.load_from_checkpoint(
                    model_filename_C, map_location=device, **loaded_model_params
                )

                architecture_params = set(model_params.keys()) - set(
                    loaded_model_params.keys()
                )

                for param in architecture_params:
                    if model_params[param] != self.Transformer_N.hparams[param]:
                        warnings.warn(
                            f"Mismatching {param} parameter in "
                            f"model checkpoint ({self.Transformer_N.hparams[param]}) "
                            f"vs config file ({model_params[param]}); "
                            "using the checkpoint."
                        )

                    if model_params[param] != self.Transformer_C.hparams[param]:
                        warnings.warn(
                            f"Mismatching {param} parameter in "
                            f"model checkpoint ({self.Transformer_C.hparams[param]}) "
                            f"vs config file ({model_params[param]}); "
                            "using the checkpoint."
                        )
            except RuntimeError:
                try:
                    self.Transformer_N = Transformer.load_from_checkpoint(
                        model_filename_N,
                        map_location=device,
                        **model_params,
                    )
                    self.Transformer_C = Transformer.load_from_checkpoint(
                        model_filename_C,
                        map_location=device,
                        **model_params,
                    )
                except RuntimeError:
                    raise RuntimeError(
                        "Weights file incompatible with the current version of Casanovo."
                    )
        else:
            sys.exit(0)

    def _get_strategy(self) -> Union[str, DDPStrategy]:
        """Get the strategy for the Trainer.

        The DDP strategy works best when multiple GPUs are used. It can work
        for CPU-only, but definitely fails using MPS (the Apple Silicon chip)
        due to Gloo.

        Returns
        -------
        Union[str, DDPStrategy]
            The strategy parameter for the Trainer.

        """
        if self._configs['Model']['Trainer'] in ("cpu", "mps"):
            return "auto"
        elif self._configs['Model']['Trainer']['devices'] == 1:
            return "auto"
        elif torch.cuda.device_count() > 1:
            return DDPStrategy(find_unused_parameters=False, static_graph=True)
        else:
            return "auto"
