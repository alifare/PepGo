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
from concurrent.futures import ProcessPoolExecutor, as_completed

import threading
from concurrent.futures import ThreadPoolExecutor

import lightning
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from .Transformer import Transformer
from .MCTTS import Monte_Carlo_Double_Root_Tree
from .utils import UTILS
from .HDF import HDF
from pprint import pprint

from pathlib import Path
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

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

torch.multiprocessing.set_sharing_strategy('file_system')

#from torch.multiprocessing import Process, Manager, Pool

from .utils import UTILS

class GPUWorker:
    def __init__(self, meta, configs, gpu_idx, model_N, model_C, inner_max_workers=None, mode=0, delta=-1, monitor=None):
        #self.monte = monte
        self.gpu_idx = gpu_idx
        self.device = torch.device(f'cuda:{gpu_idx}')
        self.mode = mode
        self.delta = delta
        self._utils = UTILS()
        self.monitor = monitor

        # 初始化统计信息
        self.stats = {
            'total_batches': 0,
            'total_samples': 0,
            'failed_samples': 0,
            'start_time': time.time(),
            'batch_times': [],
            'processing_times': []
        }
        print(f'初始化设备: {self.device}')

        # 设置内部线程池 - 用于并行处理单个批次内的样本
        #self.inner_max_workers = inner_max_workers or min(16, torch.cuda.device_count() * 2)
        #self.thread_pool = ThreadPoolExecutor(max_workers=self.inner_max_workers)

        # 模型副本
        with torch.cuda.device(self.device):
            self.model_N = copy.deepcopy(model_N).to(self.device)
            self.model_C = copy.deepcopy(model_C).to(self.device)
            self.model_N.eval()
            self.model_C.eval()

        '''
        self.monte = Monte_Carlo_Double_Root_Tree(
            meta=meta, configs=configs,
            Transformer_N=self.model_N,
            Transformer_C=self.model_C
        )
        '''

        self.monte = Monte_Carlo_Double_Root_Tree(meta=meta, configs=configs)

    def inference(self, batch_data):
        """
        推理函数 - 正确处理并行
        """
        if self.monitor:
            self.monitor.track_gpu_activity(self.gpu_idx, True)

        batch_start = time.time()
        try:
            # 1. 批量推理
            with torch.no_grad(), torch.cuda.device(self.device):
                N_memories, N_mem_masks, precursors, peptides = self.model_N(batch_data)
                C_memories, C_mem_masks, _, _ = self.model_C(batch_data)

            batch_size = N_memories.shape[0]
            self.stats['total_batches'] += 1
            self.stats['total_samples'] += batch_size

            samples = [
                N_memories.detach(),
                N_mem_masks.detach(),
                C_memories.detach(),
                C_mem_masks.detach(),
                precursors.detach(),
                peptides,
                self.mode,
                self.delta
            ]
            results = self.monte.UCTSEARCH_final(samples, self.model_N, self.model_C)

            # 记录处理时间
            batch_time = time.time() - batch_start
            self.stats['batch_times'].append(batch_time)
            return results

        except Exception as e:
            print(f"GPU{self.gpu_idx} 推理错误: {e}")
            raise

        finally:
            # 通知监控器结束
            if self.monitor:
                self.monitor.track_gpu_activity(self.gpu_idx, False)

    def get_stats(self):
        """
        获取统计信息
        """
        total_samples = self.stats['total_samples']
        failed_samples = self.stats['failed_samples']

        # 计算成功率
        if total_samples > 0:
            success_rate = (total_samples - failed_samples) / total_samples * 100
        else:
            success_rate = 0

        # 计算平均时间
        if len(self.stats['batch_times']) > 0:
            avg_batch_time = sum(self.stats['batch_times']) / len(self.stats['batch_times'])
        else:
            avg_batch_time = 0

        if total_samples > 0:
            avg_time_per_sample = sum(self.stats['processing_times']) / total_samples
        else:
            avg_time_per_sample = 0

        # 计算总运行时间
        total_runtime = time.time() - self.stats['start_time']

        return {
            'total_samples': total_samples,
            'failed_samples': failed_samples,
            'success_rate': success_rate,
            'avg_batch_time': avg_batch_time,
            'avg_time_per_sample': avg_time_per_sample,
            'total_batches': self.stats['total_batches'],
            'total_runtime': total_runtime,
            'samples_per_second': total_samples / max(0.001, total_runtime)
        }

    def print_detailed_stats(self):
        """打印详细统计信息"""
        stats = self.get_stats()

        print(f"\n{'=' * 60}")
        print(f"GPU{self.gpu_idx} 详细统计")
        print(f"{'=' * 60}")
        print(f"总批次处理: {stats['total_batches']}")
        print(f"总样本处理: {stats['total_samples']}")
        print(f"失败样本数: {stats['failed_samples']}")
        print(f"成功率: {stats['success_rate']:.2f}%")
        print(f"平均批次时间: {stats['avg_batch_time']:.3f}秒")
        print(f"平均样本时间: {stats['avg_time_per_sample']:.3f}秒")
        print(f"总运行时间: {stats['total_runtime']:.2f}秒")
        print(f"处理速度: {stats['samples_per_second']:.2f} 样本/秒")

        # 显示处理时间分布
        if self.stats['batch_times']:
            print(f"\n批次时间分布:")
            print(f"  最短: {min(self.stats['batch_times']):.3f}秒")
            print(f"  最长: {max(self.stats['batch_times']):.3f}秒")
            print(f"  中位数: {sorted(self.stats['batch_times'])[len(self.stats['batch_times']) // 2]:.3f}秒")

        if self.stats['processing_times']:
            print(f"\n样本处理时间分布:")
            print(f"  最短: {min(self.stats['processing_times']):.3f}秒")
            print(f"  最长: {max(self.stats['processing_times']):.3f}秒")

    def inference_async(self, batch_data):
        """
        异步推理版本（不等待结果）
        """
        return self.thread_pool.submit(self.inference, batch_data)

    def cleanup(self):
        """清理资源"""
        #self.thread_pool.shutdown(wait=True)

        # 清理GPU内存
        del self.model_N, self.model_C, self.monte
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

        # 打印最终统计
        self.print_detailed_stats()

        print(f"GPU{self.gpu_idx} 资源已清理")

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

        self.num_GPUs = torch.cuda.device_count()
        self.num_CPUs = os.cpu_count()

        self.isotope_error_range = self._configs['MCTTS']['Tree']['isotope_error_range']

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

            #int_array = torch.sqrt(s[:,1])
            #int_array /= torch.linalg.norm(int_array)
            #s[:,1] = int_array

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
        #Training self.Transformer_N
        train_spec_set = HDF(train_spec)
        train_spec_set_loader = torch.utils.data.DataLoader(
            train_spec_set,
            batch_size=self._configs['Model']['Trainer']['train_batch_size'],
            num_workers=self._configs['Model']['Trainer']['min_workers'],
            collate_fn=self.spec_collate,
            shuffle=True,
        )
        valid_spec_set_loader = None
        if(valid_spec is not None):
            valid_spec_set = HDF(valid_spec)
            valid_spec_set_loader = torch.utils.data.DataLoader(
                valid_spec_set,
                batch_size=self._configs['Model']['Trainer']['valid_batch_size'],
                num_workers=self._configs['Model']['Trainer']['min_workers'],
                collate_fn=self.spec_collate
            )
        #print('Training Transformer_N ...')
        self.trainer_N.fit(self.Transformer_N, train_dataloaders=train_spec_set_loader, val_dataloaders=valid_spec_set_loader)
        #del train_spec_set, valid_spec_set

        #Training self.Transformer_C
        train_spec_set = HDF(train_spec, reverse = True)
        train_spec_set_loader = torch.utils.data.DataLoader(
            train_spec_set,
            batch_size=self._configs['Model']['Trainer']['train_batch_size'],
            num_workers=self._configs['Model']['Trainer']['min_workers'],
            collate_fn=self.spec_collate,
            shuffle=True,
        )
        valid_spec_set_loader = None
        if(valid_spec is not None):
            valid_spec_set = HDF(valid_spec, reverse=True)
            valid_spec_set_loader = torch.utils.data.DataLoader(
                valid_spec_set,
                batch_size=self._configs['Model']['Trainer']['valid_batch_size'],
                num_workers=self._configs['Model']['Trainer']['min_workers'],
                collate_fn=self.spec_collate
            )
        #print('Training Transformer_C ...')
        self.trainer_C.fit(self.Transformer_C, train_dataloaders=train_spec_set_loader, val_dataloaders=valid_spec_set_loader)
        #del train_spec_set, valid_spec_set

    def predict(self, spec_file=None):
        mp.set_start_method('fork', force=True)

        out_file = os.path.basename(spec_file) \
                +'.depth'+str(self._configs['MCTTS']['Tree']['depth']) \
                +'.probe_layers'+str(self._configs['MCTTS']['Tree']['probe_layers']) \
                +'.depth_Transformer'+str(self._configs['MCTTS']['Tree']['depth_Transformer']) \
                +'.depth_Transformer_beam'+str(self._configs['MCTTS']['Tree']['depth_Transformer_beam']) \
                +'.ceiling'+str(self._configs['MCTTS']['Delta']['ceiling']) \
                +'.budget'+str(self._configs['MCTTS']['Tree']['budget']) \
                +'.isotope_error_range'+'-'.join([str(i) for i in self.isotope_error_range]) \
                +'.T_beam_search'+str(int(self._configs['MCTTS']['Delta']['mode']['transformer_beam_search'])) \
                +'.beam'+str(self._configs['Model']['Transformer']['n_beams']) \
                +'.gap_mass.result.txt'
        f_out = open(out_file,'w')
        f_out.write('#true_peptide\tpred_peptide\tmatched\ttrue_mass\tpred_mass\tmass_error\n')
        #f_out.write('#true_peptide\tpred_peptide\tmatched\ttrue_mass\tpred_mass\tmass_error\t')
        #f_out.write('probe\tT_bisect\tT_beam\n')

        start_time = time.time()

        num_workers = 4
        spec_set = HDF(spec_file)
        spec_set_loader = torch.utils.data.DataLoader(
            spec_set,
            batch_size=self._configs['Model']['Trainer']['test_batch_size'],
            num_workers= num_workers,
            collate_fn=self.spec_collate,
            shuffle=False,
            persistent_workers=False
        )

        print(f"CPU核心数: {os.cpu_count()}")
        print(f"使用 {self.num_GPUs} 个GPU进行推理")
        print(f"DataLoader共有 {len(spec_set_loader)} 个批次")

        # 创建GPU workers
        gpu_workers = []
        for i in range(self.num_GPUs):
            worker = GPUWorker(self._meta, self._configs, i, self.Transformer_N, self.Transformer_C, mode=0, delta=-2)
            gpu_workers.append(worker)

        for batch_idx, batch_data in enumerate(spec_set_loader):
            batch_results = gpu_workers[0].inference(batch_data)
            for result in batch_results:
                if (result is not None):
                    line = '\t'.join([str(i) for i in result])
                else:
                    line = str(result)
                f_out.write(line + '\n')
        f_out.close()

        # 清理GPU内存
        print("\n清理资源...")
        for worker in gpu_workers:
            try:
                worker.cleanup()
            except Exception as e:
                print(f"清理GPU{worker.gpu_idx}时出错: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("推理完成！")

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
        elif(mode=='predict'):
            devices = "auto"
            additional_cfg = dict(
                devices=devices,
                accelerator="auto",
                enable_progress_bar=True,
                strategy="auto"
                #strategy="ddp_spawn"
                #strategy = self._get_strategy()
            )
            trainer_cfg.update(additional_cfg)
        else:
            raise ValueError('The mode must be train or predict!')

        #self._utils.parse_var(trainer_cfg)

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
            out_writer=self.writer,
            meta=self._meta
        )

        loaded_model=None
        if(mode=='train'):
            Transformer_model = Transformer(**model_params)
            return(Transformer_model)
        elif(mode=='predict'):
            ckpt_file = os.path.join(model_dir, 'checkpoints', 'last.ckpt')
            if(not os.path.exists(ckpt_file)):
                raise ValueError('Please check the directory of Transormer models!')
            #self._utils.parse_var(device)
            device='cpu'
            try:
                loaded_model = Transformer.load_from_checkpoint(
                    ckpt_file, map_location=device, **loaded_model_params
                )

                architecture_params = set(model_params.keys()) - set(
                    loaded_model_params.keys()
                )
                for param in architecture_params:
                    if model_params[param] != loaded_model.hparams[param]:
                        warnings.warn(
                            f"Mismatching {param} parameter in "
                            f"model checkpoint ({loaded_model.hparams[param]}) "
                            f"vs config file ({model_params[param]}); "
                            "using the checkpoint."
                        )
            except RuntimeError:
                try:
                    loaded_model = Transformer.load_from_checkpoint(
                        ckpt_file,
                        map_location=device,
                        **model_params,
                    )
                except RuntimeError:
                    raise RuntimeError(
                        "Weights file incompatible with the current version of PepGo."
                    )
        else:
            sys.exit(0)

        return(loaded_model)

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
