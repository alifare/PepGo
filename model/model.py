#The development began around 2019-02-21
import os
import sys

import time
from datetime import datetime

import pandas as pd
import warnings
import numpy as np
np.set_printoptions(suppress=True)

from tqdm import tqdm
from typing import Union

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Manager, Pool

import lightning
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from .Transformer import Transformer
from .MCTTS import Monte_Carlo_Double_Root_Tree
from .utils import UTILS
from .HDF import HDF
from pprint import pprint

import logging
logger = logging.getLogger("PepGo")

import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*Checkpoint directory .* exists and is not empty.*",
    category=UserWarning,
    module="lightning.pytorch.callbacks.model_checkpoint"
)
print("✅ 已禁用Checkpoint路径已存在警告")

class MODEL:
    def __init__(self, meta, configs):
        super().__init__()
        self._meta = meta
        self._proton = self._meta.proton
        self._configs = configs
        self._utils = UTILS()

        # Initialized later:
        self.tmp_dir = None
        self.trainer_N = None
        self.trainer_C = None
        self.Transformer_N = None
        self.Transformer_C = None

        self.loaders = None
        self.writer = None

        self.current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')

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
        '''
        print('Train set',end=':')
        print(train_spec)
        print('Valid set',end=':')
        print(valid_spec)
        '''

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
        #print('Training Transformer_N ...')
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
        #print('Training Transformer_C ...')
        #self.trainer_C.fit(self.Transformer_C, train_dataloaders=train_spec_set_loader, val_dataloaders=valid_spec_set_loader)
        del train_spec_set, valid_spec_set

    def predict(self, spec_file=None):
        self.initialize_trainer(train=False)
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

        #spec_set = SpecDataSet(spec_file, False)
        spec_set = HDF(spec_file)
        spec_set_loader = torch.utils.data.DataLoader(
            spec_set,
            batch_size=self._configs['Model']['Trainer']['test_batch_size'],
            num_workers=self._configs['Model']['Trainer']['num_workers'],
            collate_fn=self.spec_collate,
        )

        N_memories, N_mem_masks, N_precursors, N_peptides = self.trainer_N.predict(model=self.Transformer_N, dataloaders=spec_set_loader)
        #C_memories, C_mem_masks, C_precursors, C_peptides = self.trainer_C.predict(model=self.Transformer_C, dataloaders=spec_set_loader)
        #print('N_memories',end=':')
        #print(N_memories.shape)

        sys.exit()

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

    def configure_callbacks(self, model_dir: str):
        curr_filename = self.current_datetime + "-{epoch:02d}-{step}-{valid_CELoss:.3f}"
        checkpoints_dir = os.path.join(model_dir, 'checkpoints')

        # 2. 历史快照（验证损失排序），可留多个
        hist_cb = ModelCheckpoint(
            filename=curr_filename,
            every_n_epochs=1,
            dirpath=checkpoints_dir,
            monitor="valid_CELoss",
            mode="min",
            save_top_k=self._configs['Model']['Trainer']['save_top_k'],
            save_last='link',  # Added by ChangYuqi
            enable_version_counter=False,  # Added by ChangYuqi
        )
        callbacks = [hist_cb]

        return(callbacks)

    def initialize_trainer(self, mode: str, model_dir: str) -> None:
        trainer_cfg = dict(
            accelerator=self._configs['Model']['Trainer']['accelerator'],
            devices=self._configs['Model']['Trainer']['devices'],
            enable_checkpointing=False,
            precision=self._configs['Model']['Trainer']['precision'],
            logger=False,
        )

        if(mode=='train'):
            devices = "auto" if(self._configs['Model']['Trainer']['devices'] is None) else self._configs['Model']['Trainer']['devices']
            callbacks = self.configure_callbacks(model_dir=model_dir)

            # Configure loggers.
            loggers = []

            if self._configs['Model']['Trainer']['log_metrics']:
                loggers.append(
                    lightning.pytorch.loggers.CSVLogger(
                        save_dir=model_dir, version=self.current_datetime, name="csv_logs"
                    )
                )

            if self._configs['Model']['Trainer']['tb_summarywriter']:
                loggers.append(
                    lightning.pytorch.loggers.TensorBoardLogger(
                        save_dir=model_dir, version=self.current_datetime, name="tensorboard"
                    )
                )

            if len(loggers) > 0:
                callbacks.append(
                    LearningRateMonitor(
                        log_momentum=True, log_weight_decay=True
                    ),
                )

            additional_cfg = dict(
                devices=devices,
                max_epochs=self._configs['Model']['Trainer']['max_epochs'],
                num_sanity_val_steps=self._configs['Model']['Trainer']['num_sanity_val_steps'],
                accumulate_grad_batches=self._configs['Model']['Trainer']['accumulate_grad_batches'],
                gradient_clip_val=self._configs['Model']['Trainer']['gradient_clip_val'],
                gradient_clip_algorithm=self._configs['Model']['Trainer']['gradient_clip_algorithm'],
                callbacks=callbacks,
                check_val_every_n_epoch=self._configs['Model']['Trainer'].get('check_val_every_n_epoch', 1),
                enable_checkpointing=True,
                logger=loggers,
                strategy=self._get_strategy(),
            )

            trainer_cfg.update(additional_cfg)

            '''
            print('trainer_cfg',end=':')
            print(len(trainer_cfg))
            pprint(trainer_cfg)
            print('-'*100)
            '''


        trainer = pl.Trainer(**trainer_cfg)

        return(trainer)

    def initialize_models(self, mode=None, models_dir=None) -> None:
        model_dir_N = os.path.join(models_dir, 'ckpt_N')
        model_dir_C = os.path.join(models_dir, 'ckpt_C')
        self._utils.make_dir(model_dir_N)
        self._utils.make_dir(model_dir_C)

        self.trainer_N = self.initialize_trainer(mode=mode, model_dir=model_dir_N)
        self.trainer_C = self.initialize_trainer(mode=mode, model_dir=model_dir_C)

        self.Transformer_N = self.initialize_one_model(mode=mode, model_dir=model_dir_N)
        self.Transformer_C = self.initialize_one_model(mode=mode, model_dir=model_dir_C)

    def initialize_one_model(self, mode=None, model_dir=None) -> None:
        model_params = dict(
            precursor_mass_tol=self._configs["Model"]["Transformer"]['precursor_mass_tol'],
            isotope_error_range=tuple(self._configs['Model']['Transformer']['isotope_error_range']),
            min_peptide_len=self._configs["Model"]["Transformer"]['min_peptide_len'],
            top_match=self._configs["Model"]["Transformer"]['top_match'],
            n_beams=self._configs["Model"]["Transformer"]['n_beams'],
            n_log=self._configs["Model"]["Transformer"]['n_log'],
            max_charge=self._configs["Model"]["Transformer"]['max_charge'],
            dim_model=self._configs["Model"]["Transformer"]['dim_model'],
            n_head=self._configs["Model"]["Transformer"]['n_head'],
            dim_feedforward=self._configs["Model"]["Transformer"]['dim_feedforward'],
            n_layers=self._configs["Model"]["Transformer"]['n_layers'],
            dropout=self._configs["Model"]["Transformer"]['dropout'],
            warmup_iters=self._configs['Model']['Transformer']['warmup_iters'],
            cosine_schedule_period_iters=self._configs['Model']['Transformer']['cosine_schedule_period_iters'],
            lr=self._configs['Model']['Trainer']['learning_rate'],
            weight_decay=self._configs['Model']['Trainer']['weight_decay'],
            train_label_smoothing=self._configs['Model']['Transformer']['train_label_smoothing'],
            calculate_precision=self._configs['Model']['Transformer']['calculate_precision'],
            out_writer=self.writer,
            tokenizer=None,
            meta=self._meta
        )

        loaded_model_params = dict(
            precursor_mass_tol=self._configs['Model']['Transformer']['precursor_mass_tol'],
            isotope_error_range=self._configs['Model']['Transformer']['isotope_error_range'],
            min_peptide_len=self._configs['Model']['Transformer']['min_peptide_len'],
            max_peptide_len=self._configs["Model"]["Transformer"]['max_length'],
            top_match=self._configs['Model']['Transformer']['top_match'],
            n_beams=self._configs['Model']['Transformer']['n_beams'],
            n_log=self._configs['Model']['Transformer']['n_log'],
            warmup_iters=self._configs['Model']['Transformer']['warmup_iters'],
            cosine_schedule_period_iters=self._configs['Model']['Transformer']['cosine_schedule_period_iters'],
            lr=self._configs['Model']['Trainer']['learning_rate'],
            weight_decay=self._configs['Model']['Trainer']['weight_decay'],
            train_label_smoothing=self._configs['Model']['Transformer']['train_label_smoothing'],
            calculate_precision=self._configs['Model']['Transformer']['calculate_precision'],
            out_writer=self.writer,
            meta=self._meta
        )

        if(mode=='train'):
            Transformer_model = Transformer(**model_params)
            return(Transformer_model)

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
