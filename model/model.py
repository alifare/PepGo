#The development began around 2019-02-21
import os
import sys

import time
from datetime import datetime

import pathlib
import pandas as pd
import warnings
import numpy as np
np.set_printoptions(suppress=True)

from tqdm import tqdm
from typing import Union

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Manager, Pool

import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint


from .Transformer import Transformer
from .MCTTS import Monte_Carlo_Double_Root_Tree
from .utils import UTILS
from .HDF import HDF
import pprint as pp

class MODEL:
    def __init__(self, meta, configs):
        super().__init__()
        self._meta = meta
        self._proton = self._meta.proton
        self._configs = configs
        self._utils = UTILS()

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

    def train(self, train_spec=None, valid_spec=None, mode=None):
        print('Train set',end=':')
        print(train_spec)
        print('Valid set',end=':')
        print(valid_spec)

        #Training self.Transformer_N
        train_spec_set = HDF(train_spec)
        valid_spec_set = HDF(valid_spec)
        train_spec_set_loader = torch.utils.data.DataLoader(
            train_spec_set,
            batch_size=self._configs['Model']['Trainer']['train_batch_size'],
            num_workers=self._configs['Model']['Trainer']['num_workers'],
            collate_fn=self.spec_collate,
            shuffle=True,
        )
        valid_spec_set_loader = torch.utils.data.DataLoader(
            valid_spec_set,
            batch_size=self._configs['Model']['Trainer']['valid_batch_size'],
            num_workers=self._configs['Model']['Trainer']['num_workers'],
            collate_fn=self.spec_collate
        )
        print('Training Transformer_N ...')
        self.trainer_N.fit(self.Transformer_N, train_dataloaders=train_spec_set_loader, val_dataloaders=valid_spec_set_loader)
        del train_spec_set, valid_spec_set

        #Training self.Transformer_C
        train_spec_set = HDF(train_spec, reverse = True)
        valid_spec_set = HDF(valid_spec, reverse = True)
        train_spec_set_loader = torch.utils.data.DataLoader(
            train_spec_set,
            batch_size=self._configs['Model']['Trainer']['train_batch_size'],
            num_workers=self._configs['Model']['Trainer']['num_workers'],
            collate_fn=self.spec_collate,
            shuffle=True,
        )
        valid_spec_set_loader = torch.utils.data.DataLoader(
            valid_spec_set,
            batch_size=self._configs['Model']['Trainer']['valid_batch_size'],
            num_workers=self._configs['Model']['Trainer']['num_workers'],
            collate_fn=self.spec_collate
        )
        print('Training Transformer_C ...')
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
        trainer_cfg = dict(
            accelerator=self._configs['Model']['Trainer']['accelerator'],
            devices=1,
            enable_checkpointing=False,
            precision=self._configs['Model']['Trainer']['precision'],
            logger=False,
        )

        if train:
            if self._configs['Model']['Trainer']['devices'] is None:
                devices = "auto"
            else:
                devices = self._configs['Model']['Trainer']['devices']

            # Configure loggers.
            loggers = []
            output_dir = 'output_dir'
            overwrite_ckpt_check = 'overwrite_ckpt_check'
            if self._configs['Model']['Trainer']['log_metrics'] or self._configs['Model']['Trainer']['tb_summarywriter']:
                if not output_dir:
                    logger.warning(
                        "Output directory not set in model runner. "
                        "No loss file or Tensorboard will be created."
                    )
                else:
                    csv_log_dir = "csv_logs"
                    tb_log_dir = "tensorboard"

                    '''
                    if self._configs['Model']['Trainer']['log_metrics']:
                        if overwrite_ckpt_check:
                            utils.check_dir_file_exists(
                                output_dir, csv_log_dir
                            )

                        loggers.append(
                            lightning.pytorch.loggers.CSVLogger(
                                output_dir, version=csv_log_dir, name=None
                            )
                        )

                    if self._configs['Model']['Trainer']['tb_summarywriter']:
                        if overwrite_ckpt_check:
                            utils.check_dir_file_exists(
                                output_dir, tb_log_dir
                            )

                        loggers.append(
                            lightning.pytorch.loggers.TensorBoardLogger(
                                output_dir, version=tb_log_dir, name=None
                            )
                        )

                    if len(loggers) > 0:
                        self.callbacks.append(
                            LearningRateMonitor(
                                log_momentum=True, log_weight_decay=True
                            ),
                        )
                    '''

            additional_cfg = dict(
                devices=devices,
                val_check_interval=self._configs['Model']['Trainer']['val_check_interval'],
                max_epochs=self._configs['Model']['Trainer']['epoch'],
                num_sanity_val_steps=self._configs['Model']['Trainer']['num_sanity_val_steps'],
                accumulate_grad_batches=self._configs['Model']['Trainer']['accumulate_grad_batches'],
                gradient_clip_val=self._configs['Model']['Trainer']['gradient_clip_val'],
                gradient_clip_algorithm=self._configs['Model']['Trainer']['gradient_clip_algorithm'],
                #check_val_every_n_epoch=1,
                check_val_every_n_epoch=None,
                enable_checkpointing=True,
                logger=loggers,
                strategy=self._get_strategy(),
            )

            trainer_cfg.update(additional_cfg)

            trainer_cfg_N = trainer_cfg.copy()
            trainer_cfg_C = trainer_cfg.copy()

            trainer_cfg_N['callbacks']=self.callbacks_N
            trainer_cfg_C['callbacks']=self.callbacks_C

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
            #dim_intensity = self._configs["Model"]["Transformer"]['dim_intensity'],
            #max_length = self._configs["Model"]["Transformer"]['max_length'],
            #residues = residues,
            max_charge = self._configs["Model"]["Transformer"]['max_charge'],
            precursor_mass_tol = self._configs["Model"]["Transformer"]['precursor_mass_tol'],

            isotope_error_range = tuple(self._configs['Model']['Transformer']['isotope_error_range']),

            min_peptide_len = self._configs["Model"]["Transformer"]['min_peptide_len'],
            n_beams = self._configs["Model"]["Transformer"]['n_beams'],
            top_match = self._configs["Model"]["Transformer"]['top_match'],
            n_log = self._configs["Model"]["Transformer"]['n_log'],
            #tb_summarywriter =self._configs['Model']['Transformer']['tb_summarywriter'],
            train_label_smoothing=self._configs['Model']['Transformer']['train_label_smoothing'],
            warmup_iters=self._configs['Model']['Transformer']['warmup_iters'],
            cosine_schedule_period_iters=self._configs['Model']['Transformer']['cosine_schedule_period_iters'],
            #max_iters=self._configs['Model']['Transformer']['max_iters'],
            lr=self._configs['Model']['Trainer']['learning_rate'],
            weight_decay=self._configs['Model']['Trainer']['weight_decay'],
            out_writer=self.writer,
            calculate_precision=self._configs['Model']['Transformer']['calculate_precision'],
            tokenizer=None,
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

            prefix = f"{datetime.now().strftime('%Y%m%d_%H%M%S_')}"

            curr_filename = prefix + "{epoch}-{step}"
            best_filename = prefix + "best"
            # Configure checkpoints.
            self.callbacks_N = [
                ModelCheckpoint(
                    dirpath=models_dir_N,
                    save_on_train_epoch_end=True,
                    filename=curr_filename,
                    save_top_k=self._configs['Model']['Trainer']['save_top_k'],
                    save_last='link',  # Added by ChangYuqi
                    enable_version_counter=False,
                ),
                ModelCheckpoint(
                    dirpath=models_dir_N,
                    monitor="valid_CELoss",
                    filename=best_filename,
                    save_top_k=self._configs['Model']['Trainer']['save_top_k'],
                    save_last='link',  # Added by ChangYuqi
                    enable_version_counter=False,
                ),
            ]

            self.callbacks_C = [
                ModelCheckpoint(
                    dirpath=models_dir_C,
                    save_on_train_epoch_end=True,
                    filename=curr_filename,
                    save_top_k=self._configs['Model']['Trainer']['save_top_k'],
                    save_last='link',  # Added by ChangYuqi
                    enable_version_counter=False,
                ),
                ModelCheckpoint(
                    dirpath=models_dir_C,
                    monitor="valid_CELoss",
                    filename=best_filename,
                    #mode="min",
                    save_top_k=self._configs['Model']['Trainer']['save_top_k'],
                    save_last='link',  # Added by ChangYuqi
                    enable_version_counter=False,
                ),
            ]
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
