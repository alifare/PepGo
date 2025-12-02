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

from .utils import UTILS

class GPUWorker:
    def __init__(self, meta, configs, gpu_idx, model_N, model_C, inner_max_workers=None, mode=0, delta=-1):
        self.gpu_idx = gpu_idx
        self.device = torch.device(f'cuda:{gpu_idx}')
        self.mode = mode
        self.delta = delta
        self._utils = UTILS()

        print(f'初始化设备: {self.device}')

        # 设置内部线程池 - 用于并行处理单个批次内的样本
        self.inner_max_workers = inner_max_workers or min(4, torch.cuda.device_count() * 2)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.inner_max_workers)

        # 使用锁保护模型访问（如果模型本身不是线程安全的）
        self.model_lock = threading.Lock()

        # 模型副本
        with torch.cuda.device(self.device):
            self.model_N = copy.deepcopy(model_N).to(self.device)
            self.model_C = copy.deepcopy(model_C).to(self.device)
            self.model_N.eval()
            self.model_C.eval()

        self.monte = Monte_Carlo_Double_Root_Tree(
            meta=meta, configs=configs,
            Transformer_N=self.model_N,
            Transformer_C=self.model_C
        )

    def inference(self, batch_data):
        """
        推理函数 - 正确处理并行
        """
        try:
            # 1. 数据移动到GPU（这部分很快，不需要锁）
            #batch_data = self._move_to_device(batch_data)
            # 2. 批量推理（这部分需要模型锁）
            with self.model_lock, torch.no_grad(), torch.cuda.device(self.device):
                N_memories, N_mem_masks, precursors, peptides = self.model_N(batch_data)
                C_memories, C_mem_masks, _, _ = self.model_C(batch_data)

            # 3. 并行处理样本（这部分不需要锁，可以并行）
            return self._parallel_process_samples(
                N_memories, N_mem_masks, C_memories, C_mem_masks, precursors, peptides
            )

        except Exception as e:
            print(f"GPU{self.gpu_idx} 推理错误: {e}")
            raise

    def _move_to_device(self, batch_data):
        """将数据移动到正确的GPU"""
        if isinstance(batch_data, (list, tuple)):
            return [
                item.to(self.device, non_blocking=True)
                if torch.is_tensor(item) else item
                for item in batch_data
            ]
        elif torch.is_tensor(batch_data):
            return batch_data.to(self.device, non_blocking=True)
        else:
            return batch_data

    def _parallel_process_samples(self, N_memories, N_mem_masks, C_memories, C_mem_masks, precursors, peptides):
        """
        并行处理批次内的所有样本
        """
        batch_size = N_memories.shape[0]
        results = [None for _ in range(batch_size)]
        futures = {}

        # 提交所有样本处理任务到线程池
        for i in range(batch_size):
            future = self.thread_pool.submit(
                self._process_single_sample,
            i, N_memories, N_mem_masks, C_memories, C_mem_masks, precursors, peptides
            )
            futures[future] = i

        # 收集结果 - 使用as_completed提高效率
        completed = 0
        start_time = time.time()

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result(timeout=300)  # 5分钟超时
                results[idx] = result
                completed += 1

                # 打印进度
                if completed % 10 == 0 or completed == batch_size:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed if completed > 0 else 0
                    remaining = (batch_size - completed) * avg_time if avg_time > 0 else 0

                    print(f"GPU{self.gpu_idx}: {completed}/{batch_size} "
                          f"({completed / batch_size * 100:.1f}%), "
                          f"预计剩余: {remaining:.1f}s")

            except TimeoutError:
                print(f"GPU{self.gpu_idx} 样本 {idx} 处理超时")
                results[idx] = None
            except Exception as e:
                print(f"GPU{self.gpu_idx} 样本 {idx} 处理错误: {e}")
                results[idx] = None

        return results

    def _process_single_sample(self, idx, N_memories, N_mem_masks, C_memories, C_mem_masks, precursors, peptides):
        """
        处理单个样本
        """
        try:
            # 提取样本数据
            sample_data = [
                N_memories[idx:idx + 1].detach(),
                N_mem_masks[idx:idx + 1].detach(),
                C_memories[idx:idx + 1].detach(),
                C_mem_masks[idx:idx + 1].detach(),
                precursors[idx:idx + 1].detach(),
                peptides[idx:idx + 1],
                self.mode,
                self.delta
            ]
            # 执行Monte Carlo搜索
            result = self.monte.UCTSEARCH_Transformer(sample_data)

            return result

        except Exception as e:
            print(f"GPU{self.gpu_idx} 样本 {idx} 处理错误: {e}")
            return None

    def inference_async(self, batch_data):
        """
        异步推理版本（不等待结果）
        """
        return self.thread_pool.submit(self.inference, batch_data)

    def cleanup(self):
        """清理资源"""
        self.thread_pool.shutdown(wait=True)

        # 清理GPU内存
        del self.model_N, self.model_C, self.monte
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

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
        f_out.write('#true_peptide\tpred_peptide\tmatched\ttrue_mass\tpred_mass\tmass_error\n')
        #f_out.write('#true_peptide\tpred_peptide\tmatched\ttrue_mass\tpred_mass\tmass_error\t')
        #f_out.write('probe\tT_bisect\tT_beam\n')

        start = time.time()
        num_gpus = torch.cuda.device_count()
        spec_set = HDF(spec_file)
        spec_set_loader = torch.utils.data.DataLoader(
            spec_set,
            batch_size=self._configs['Model']['Trainer']['test_batch_size'],
            num_workers=num_gpus,
            #num_workers=os.cpu_count() // num_gpus,
            #num_workers=self._configs['Model']['Trainer']['num_workers'],
            collate_fn=self.spec_collate,
            persistent_workers=False
        )

        print(f"CPU核心数: {os.cpu_count()}")
        print(f"使用 {num_gpus} 个GPU进行推理")
        print(f"DataLoader共有 {len(spec_set_loader)} 个批次")

        # 创建GPU workers
        gpu_workers = []
        for i in range(num_gpus):
            worker = GPUWorker(self._meta, self._configs, i, self.Transformer_N, self.Transformer_C, mode=0, delta=-2)
            gpu_workers.append(worker)

        # 使用ThreadPoolExecutor
        total_results = []
        batch_info = {}  # 记录批次信息
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            #futures = []
            futures = {}
            for batch_idx, batch_data in enumerate(spec_set_loader):
                gpu_idx = batch_idx % num_gpus
                batch_size = batch_data[0].shape[0]

                future = executor.submit(gpu_workers[gpu_idx].inference, batch_data)
                futures[future] = {
                    'batch_idx': batch_idx,
                    'gpu_idx': gpu_idx,
                    'batch_size': batch_size,
                    'submit_time': time.time()
                }

                # 打印提交进度
                if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                    print(f"已提交 {batch_idx + 1}/{len(spec_set_loader)} 个批次到 GPU{gpu_idx}")

            print(f"所有 {len(futures)} 个批次已提交，开始处理...")


            '''
            # 收集所有结果
            for future in futures:
                try:
                    batch_results = future.result(timeout=3000)  # 5分钟超时
                    total_results += batch_results
                except Exception as e:
                    print(f"GPU{gpu_idx} 批次样本处理超时或错误: {e}")
                    total_results.append(None)
            '''

            # 使用as_completed收集结果（更高效）
            completed_batches = 0
            failed_batches = 0

            for future in as_completed(futures):
                info = futures[future]
                batch_idx = info['batch_idx']
                gpu_idx = info['gpu_idx']
                expected_size = info['batch_size']

                try:
                    # 获取批次结果
                    batch_results = future.result(timeout=300)  # 50分钟超时

                    # 验证结果数量
                    if batch_results is None:
                        print(f"警告: GPU{gpu_idx} 批次 {batch_idx} 返回None")
                        batch_results = [None for _ in range(expected_size)]
                    elif len(batch_results) != expected_size:
                        print(f"警告: GPU{gpu_idx} 批次 {batch_idx} 结果数量不匹配 "
                              f"(期望 {expected_size}, 实际 {len(batch_results)})")
                        # 填充或截断结果
                        if len(batch_results) < expected_size:
                            batch_results.extend([None for _ in range(expected_size - len(batch_results))])
                        else:
                            batch_results = batch_results[:expected_size]

                    # 添加到总结果
                    total_results.extend(batch_results)
                    completed_batches += 1

                    # 计算并显示进度
                    elapsed = time.time() - start_time
                    avg_time_per_batch = elapsed / completed_batches if completed_batches > 0 else 0
                    remaining_batches = len(futures) - completed_batches
                    estimated_remaining = remaining_batches * avg_time_per_batch if avg_time_per_batch > 0 else 0

                    # 格式化的时间显示
                    def format_time(seconds):
                        hours = int(seconds // 3600)
                        minutes = int((seconds % 3600) // 60)
                        secs = int(seconds % 60)
                        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

                    print(f"✓ 批次 {batch_idx:4d} (GPU{gpu_idx}) 完成: "
                          f"{len(batch_results)} 个结果 | "
                          f"进度: {completed_batches}/{len(futures)} "
                          f"({completed_batches / len(futures) * 100:.1f}%) | "
                          f"已用: {format_time(elapsed)} | "
                          f"预计剩余: {format_time(estimated_remaining)}")

                except TimeoutError:
                    failed_batches += 1
                    print(f"✗ GPU{gpu_idx} 批次 {batch_idx} 处理超时 (5分钟)")
                    # 添加与批次大小匹配的None列表
                    total_results.extend([None] * expected_size)

                except Exception as e:
                    failed_batches += 1
                    print(f"✗ GPU{gpu_idx} 批次 {batch_idx} 处理错误: {e}")
                    # 添加与批次大小匹配的None列表
                    total_results.extend([None] * expected_size)

        # 最终统计
        end_time = time.time()
        total_time = end_time - start_time
        total_samples = len(total_results)
        successful_samples = sum(1 for r in total_results if r is not None)

        print("\n" + "=" * 60)
        print("推理完成！")
        print("=" * 60)
        print(f"总批次: {len(futures)}")
        print(f"完成批次: {completed_batches}")
        print(f"失败批次: {failed_batches}")
        print(f"成功率: {completed_batches / len(futures) * 100:.1f}%")
        print(f"总样本数: {total_samples}")
        print(f"成功样本: {successful_samples}")
        print(f"样本成功率: {successful_samples / total_samples * 100:.1f}%")
        print(f"总耗时: {format_time(total_time)}")
        print(f"平均速度: {total_samples / max(total_time, 0.001):.2f} 样本/秒")

        # 打印每个GPU的统计信息
        print("\n各GPU统计:")
        for i, worker in enumerate(gpu_workers):
            try:
                stats = worker.get_stats()
                print(f"GPU{i}: "
                      f"处理样本 {stats.get('total_samples', 0)} | "
                      f"成功率 {stats.get('success_rate', 0) * 100:.1f}% | "
                      f"平均时间 {stats.get('avg_time_per_sample', 0):.3f}s/样本")
            except:
                print(f"GPU{i}: 统计信息不可用")

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

        for result in total_results:
            if(result is not None):
                line = '\t'.join([str(i) for i in result])
                f_out.write(line+'\n')
        f_out.close()

        return total_results

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
                        "Weights file incompatible with the current version of Casanovo."
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
