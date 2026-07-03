#The development began around 2019-02-21
import os
import sys
import glob

import time
from datetime import datetime

import numpy as np
np.set_printoptions(suppress=True)

from typing import Union
import torch
import torch.multiprocessing as mp

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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

torch.multiprocessing.set_sharing_strategy('file_system')

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

        # 模型副本
        with torch.cuda.device(self.device):
            self.model_N = copy.deepcopy(model_N).to(self.device)
            self.model_C = copy.deepcopy(model_C).to(self.device)
            self.model_N.eval()
            self.model_C.eval()

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

    def cleanup(self):
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
        self.Trainer_configs = self._configs.get('Model', {}).get('Trainer', {})
        self.Pretrain_configs = self._configs.get('Model', {}).get('Pretrain', {})
        self.Basic_configs = self._configs.get('Model', {}).get('Basic', {})

        self.max_peaks = self.Basic_configs.get('max_peaks',300)

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
        charges = []
        precursors = []

        for i in item:
            s=torch.tensor(i[0])
            if(self.max_peaks>0):
                # 按强度降序排序
                intensity = s[:, 1]
                sort_idx = torch.argsort(intensity, descending=True)
                s_sorted = s[sort_idx]
                # 截断
                s_truncated = s_sorted[:self.max_peaks]
                # 重新按 m/z 升序排序
                mz_order = torch.argsort(s_truncated[:, 0])
                s = s_truncated[mz_order]

            self.normalize_intensity=True
            if(self.normalize_intensity):
                intensity = s[:, 1]
                max_intensity = intensity.max()
                if max_intensity > 0:
                    s[:, 1] = intensity / max_intensity
                else:
                    raise ValueError('max_intensity must be >0')

            #int_array = torch.sqrt(s[:,1])
            #int_array /= torch.linalg.norm(int_array)
            #s[:,1] = int_array

            prec_mass = i[2][0]
            prec_charge = i[3][0]
            prec_mz = (prec_mass / prec_charge) + self._proton
            prec_intensity = 1.1  # DreaMS 论文固定值
            prec_token = torch.tensor([[prec_mz, prec_intensity]], dtype=s.dtype)

            #self._utils.parse_var(s)
            s = torch.cat([prec_token, s], dim=0)
            #self._utils.parse_var(s)

            spectra.append(s)
            peptides.append(i[1])
            total_mass.append(i[2])
            charges.append(i[3])

            precursors.append([prec_mass, prec_charge, prec_mz])

        spectra = torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True)
        precursors = torch.tensor(precursors)
        charges = torch.tensor(charges)

        #batch = [spectra, precursors, peptides]
        batch = [spectra, precursors, peptides, charges]

        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)
        return(batch)

    def pretrain(self, pretrain_spec=None, prevalid_spec=None):
        #Training self.Transformer_N
        print('pretrain_spec',end=':')
        print(pretrain_spec)
        print('prevalid_spec',end=':')
        print(prevalid_spec)
        pretrain_spec_set = HDF(pretrain_spec)
        train_spec_set_loader = torch.utils.data.DataLoader(
            pretrain_spec_set,
            batch_size=self.Pretrain_configs.get('pretrain_batch_size'),
            num_workers=self.Pretrain_configs.get('min_workers'),
            collate_fn=self.spec_collate,
            shuffle=True,
        )

        valid_spec_set_loader = None
        if(prevalid_spec is not None):
            prevalid_spec_set = HDF(prevalid_spec)
            valid_spec_set_loader = torch.utils.data.DataLoader(
                prevalid_spec_set,
                batch_size=self.Pretrain_configs.get('prevalid_batch_size'),
                num_workers=self.Pretrain_configs.get('min_workers'),
                collate_fn=self.spec_collate
            )

        self.pretrainer.fit(self.Transformer_encoder, train_dataloaders=train_spec_set_loader, val_dataloaders=valid_spec_set_loader)
        if self.pretrainer.is_global_zero:
            best_src = self.pretrainer.checkpoint_callback.best_model_path
            if best_src:
                best_link = os.path.join(os.path.dirname(best_src), "best.ckpt")
                if os.path.lexists(best_link):
                    os.remove(best_link)
                os.symlink(os.path.basename(best_src), best_link)

    def train(self, train_spec=None, valid_spec=None, pretrained_ckpt=None, model_N=None, model_C=None):
        if pretrained_ckpt:
            print(f"Loading pretrained weights from {pretrained_ckpt}")
            # 加载检查点
            checkpoint = torch.load(pretrained_ckpt, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('state_dict', checkpoint)
            keys_N = set(self.Transformer_N.state_dict().keys())
            keys_C = set(self.Transformer_C.state_dict().keys())
            # 检查是否完全相同
            if keys_N == keys_C:
                print("✅ Keys are identical! Models have the same structure.")
            else:
                raise ValueError('❌ Keys are different!')
            # 获取模型当前状态字典
            model_state_dict = self.Transformer_N.state_dict()

            # 过滤形状匹配的键
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if key in model_state_dict and value.shape == model_state_dict[key].shape:
                    filtered_state_dict[key] = value
                else:
                    print(f"Skipping {key}: shape mismatch or not found")

            # 加载到两个模型
            self.Transformer_N.load_state_dict(filtered_state_dict, strict=False)
            self.Transformer_C.load_state_dict(filtered_state_dict, strict=False)

            print(f"✅ Loaded {len(filtered_state_dict)}/{len(model_state_dict)} parameters to both Transformer_N and Transformer_C")

        #Training self.Transformer_N
        train_spec_set = HDF(train_spec)
        train_spec_set_loader = torch.utils.data.DataLoader(
            train_spec_set,
            batch_size=self.Trainer_configs.get('train_batch_size'),
            num_workers=self.Trainer_configs.get('min_workers'),
            collate_fn=self.spec_collate,
            shuffle=True,
        )
        valid_spec_set_loader = None
        if(valid_spec is not None):
            valid_spec_set = HDF(valid_spec)
            valid_spec_set_loader = torch.utils.data.DataLoader(
                valid_spec_set,
                batch_size=self.Trainer_configs.get('valid_batch_size'),
                num_workers=self.Trainer_configs.get('min_workers'),
                collate_fn=self.spec_collate
            )
        self.trainer_N.fit(self.Transformer_N, train_dataloaders=train_spec_set_loader, val_dataloaders=valid_spec_set_loader)
        if self.trainer_N.is_global_zero:
            best_src = self.trainer_N.checkpoint_callback.best_model_path
            if best_src:
                best_link = os.path.join(os.path.dirname(best_src), "best.ckpt")
                if os.path.lexists(best_link):
                    os.remove(best_link)
                os.symlink(os.path.basename(best_src), best_link)


        #Training self.Transformer_C
        train_spec_set = HDF(train_spec, reverse = True)
        train_spec_set_loader = torch.utils.data.DataLoader(
            train_spec_set,
            batch_size=self.Trainer_configs.get('train_batch_size'),
            num_workers=self.Trainer_configs.get('min_workers'),
            collate_fn=self.spec_collate,
            shuffle=True,
        )
        valid_spec_set_loader = None
        if(valid_spec is not None):
            valid_spec_set = HDF(valid_spec, reverse=True)
            valid_spec_set_loader = torch.utils.data.DataLoader(
                valid_spec_set,
                batch_size=self.Trainer_configs.get('valid_batch_size'),
                num_workers=self.Trainer_configs.get('min_workers'),
                collate_fn=self.spec_collate
            )
        self.trainer_C.fit(self.Transformer_C, train_dataloaders=train_spec_set_loader, val_dataloaders=valid_spec_set_loader)
        if self.trainer_C.is_global_zero:
            best_src = self.trainer_C.checkpoint_callback.best_model_path
            if best_src:
                best_link = os.path.join(os.path.dirname(best_src), "best.ckpt")
                if os.path.lexists(best_link):
                    os.remove(best_link)
                os.symlink(os.path.basename(best_src), best_link)


    def predict(self, spec_file, output_spec_file=None):
        mp.set_start_method('fork', force=True)

        if(output_spec_file is None):
            output_spec_file = os.path.basename(spec_file)

        out_file = output_spec_file \
                +'.depth'+str(self._configs['MCTTS']['Tree']['depth']) \
                +'.probe_layers'+str(self._configs['MCTTS']['Tree']['probe_layers']) \
                +'.depth_Transformer'+str(self._configs['MCTTS']['Tree']['depth_Transformer']) \
                +'.depth_Transformer_beam'+str(self._configs['MCTTS']['Tree']['depth_Transformer_beam']) \
                +'.ceiling'+str(self._configs['MCTTS']['Delta']['ceiling']) \
                +'.budget'+str(self._configs['MCTTS']['Tree']['budget']) \
                +'.isotope_error_range'+'-'.join([str(i) for i in self.isotope_error_range]) \
                +'.T_beam_search'+str(int(self._configs['MCTTS']['Delta']['mode']['transformer_beam_search'])) \
                +'.gap_mass.result.txt'
        f_out = open(out_file,'w')
        f_out.write('#true_peptide\tpred_peptide\tmatched\ttrue_mass\tpred_mass\tmass_error\n')
        #f_out.write('#true_peptide\tpred_peptide\tmatched\ttrue_mass\tpred_mass\tmass_error\t')
        #f_out.write('probe\tT_bisect\tT_beam\n')

        start_time = time.time()

        #num_workers = 4
        spec_set = HDF(spec_file)
        spec_set_loader = torch.utils.data.DataLoader(
            spec_set,
            batch_size = self.Trainer_configs.get('test_batch_size'),
            num_workers= self.Trainer_configs.get('test_batch_size'),
            collate_fn=self.spec_collate,
            shuffle=False,
            persistent_workers=False
        )

        #print(f"CPU核心数: {os.cpu_count()}")
        #print(f"使用 {self.num_GPUs} 个GPU进行推理")
        print(f"DataLoader共有 {len(spec_set_loader)} 个批次")

        # 创建GPU workers
        worker = GPUWorker(self._meta, self._configs, 0, self.Transformer_N, self.Transformer_C, mode=0, delta=-2)
        for batch_idx, batch_data in enumerate(spec_set_loader):
            batch_results = worker.inference(batch_data)
            for result in batch_results:
                if (result is not None):
                    line = '\t'.join([str(i) for i in result])
                else:
                    line = str(result)
                f_out.write(line + '\n')
        f_out.close()

        try:
            worker.cleanup()
        except Exception as e:
            print(f"清理GPU{worker.gpu_idx}时出错: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("推理完成！")

    def configure_callbacks(self, mode:str =None, model_dir:str =None):
        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        hist_cb = None

        if(mode=='pretrain'):
            hist_cb = ModelCheckpoint(
                dirpath=checkpoints_dir,
                filename='pretrained-encoder-{epoch:02d}-{pretrain_val_total_loss:.4f}',
                every_n_epochs=self.Pretrain_configs.get('every_n_epochs', 1),
                monitor="pretrain_val_mz_loss",
                mode="min",
                save_top_k=self.Pretrain_configs.get('save_top_k', 1),
                enable_version_counter=False,  # Added by ChangYuqi
            )
        elif(mode=='train'):
            hist_cb = ModelCheckpoint(
                dirpath=checkpoints_dir,
                filename=self.current_datetime + "-{epoch:02d}-{step}-{valid_CELoss:.3f}",
                every_n_epochs=self.Trainer_configs.get('every_n_epochs', 1),
                monitor="valid_CELoss",
                mode="min",
                save_top_k=self.Trainer_configs.get('save_top_k', 1),
                save_last='link',  # Added by ChangYuqi
                enable_version_counter=False,  # Added by ChangYuqi
            )

        callbacks = [hist_cb]

        return(callbacks)

    def initialize_trainer(self, mode: str, model_dir: str):
        devices = self.Trainer_configs.get('devices', 'auto')

        # ===== 1. 先创建 loggers =====
        loggers = []

        if self.Trainer_configs.get('log_metrics'):
            loggers.append(
                lightning.pytorch.loggers.CSVLogger(
                    save_dir=model_dir,
                    version=self.current_datetime,
                    name="csv_logs"
                )
            )

        if self.Trainer_configs.get('tb_summarywriter'):
            loggers.append(
                lightning.pytorch.loggers.TensorBoardLogger(
                    save_dir=model_dir,
                    version=self.current_datetime,
                    name="tensorboard"
                )
            )

        # ===== 2. 创建 callbacks =====
        callbacks = self.configure_callbacks(mode=mode, model_dir=model_dir)

        if len(loggers) > 0:
            callbacks.append(
                LearningRateMonitor(log_momentum=True, log_weight_decay=True),
            )

        # ===== 3. 如果有 logger，添加 LearningRateMonitor =====
        if len(loggers) > 0:
            callbacks.append(
                LearningRateMonitor(log_momentum=True, log_weight_decay=True)
            )

        # ===== 4. 基础配置 =====
        trainer_cfg = dict(
            accelerator=self.Trainer_configs.get('accelerator'),
            devices=devices,
            enable_checkpointing=False,
            precision=self.Trainer_configs.get('precision'),
            logger=False,  # 临时值，后面会被覆盖
        )
        additional_cfg = dict()

        # ===== 5. 模式特定配置 =====
        if mode == 'pretrain':
            additional_cfg = dict(
                max_epochs=self.Pretrain_configs.get('max_epochs'),
                callbacks=callbacks,
                check_val_every_n_epoch=self.Pretrain_configs.get('check_val_every_n_epoch', 1),
                enable_checkpointing=True,
                strategy=self._get_strategy(),
                logger=loggers if loggers else False,
            )
        elif mode == 'train':
            additional_cfg = dict(
                max_epochs=self.Trainer_configs.get('max_epochs'),
                num_sanity_val_steps=self.Trainer_configs.get('num_sanity_val_steps'),
                accumulate_grad_batches=self.Trainer_configs.get('accumulate_grad_batches'),
                gradient_clip_val=self.Trainer_configs.get('gradient_clip_val'),
                gradient_clip_algorithm=self.Trainer_configs.get('gradient_clip_algorithm'),
                callbacks=callbacks,
                check_val_every_n_epoch=self.Trainer_configs.get('check_val_every_n_epoch', 1),
                enable_checkpointing=True,
                logger=loggers if loggers else False,  # 使用创建好的 loggers
                strategy=self._get_strategy(),
            )
        elif mode == 'predict':
            additional_cfg = dict(
                devices=devices,
                accelerator="auto",
                enable_progress_bar=True,
                strategy="auto"
            )
        else:
            raise ValueError('The mode must be pretrain, train, or predict!')

        # ===== 6. 合并配置 =====
        trainer_cfg.update(additional_cfg)

        # ===== 7. 创建 trainer =====
        trainer = pl.Trainer(**trainer_cfg)
        return trainer

    def initialize_models(self, mode=None, models_dir=None, prescan_spec=None) -> None:

        def prescan(data_loader):
            if data_loader is None:
                logger.warning("data_loader is None, using default schedule values")
                return

            # 计算真实的总步数
            steps_per_epoch = len(data_loader)
            max_epochs = self.Pretrain_configs.get('max_epochs') if(mode=='pretrain') else self.Trainer_configs.get('max_epochs')
            total_steps = steps_per_epoch * max_epochs

            # 根据模式选择预热比例
            if(mode=='pretrain'):
                warmup_ratio = self.Pretrain_configs.get('warmup_ratio', 0.12)
            else:
                warmup_ratio = self.Trainer_configs.get('warmup_ratio', 0.06)

            # 自动计算正确的参数
            correct_warmup_iters = int(total_steps * warmup_ratio)
            correct_cosine_period = total_steps

            if(mode=='pretrain'):
                user_warmup = self.Pretrain_configs.get('warmup_iters')
                if(user_warmup is None):
                    self.Pretrain_configs['warmup_iters'] = correct_warmup_iters

                user_cosine = self.Pretrain_configs.get('cosine_schedule_period_iters')
                if(user_cosine is None):
                    self.Pretrain_configs['cosine_schedule_period_iters'] = correct_cosine_period
            else:
                user_warmup = self.Trainer_configs.get('warmup_iters')
                if(user_warmup is None):
                    self.Trainer_configs['warmup_iters'] = correct_warmup_iters

                user_cosine = self.Trainer_configs.get('cosine_schedule_period_iters')
                if(user_cosine is None):
                    self.Trainer_configs['cosine_schedule_period_iters'] = correct_cosine_period

        if (prescan_spec is not None):
            prescan_batch_size=self.Trainer_configs.get('train_batch_size')
            prescan_min_workers=self.Trainer_configs.get('min_workers')
            if(mode=='pretrain'):
                prescan_batch_size=self.Pretrain_configs.get('pretrain_batch_size')
                prescan_min_workers=self.Pretrain_configs.get('min_workers')

            prescan_spec_set = HDF(prescan_spec)
            prescan_spec_set_loader = torch.utils.data.DataLoader(
                prescan_spec_set,
                batch_size=prescan_batch_size,
                num_workers=prescan_min_workers,
                collate_fn=self.spec_collate,
                shuffle=True,
            )

            prescan(prescan_spec_set_loader)
            del prescan_spec_set, prescan_spec_set_loader

        if(mode=='pretrain'):
            model_dir_encoder = os.path.join(models_dir, 'ckpt_pretrain')
            self._utils.make_dir(model_dir_encoder)
            self.pretrainer = self.initialize_trainer(mode=mode, model_dir=model_dir_encoder)
            self.Transformer_encoder = self.initialize_one_model(mode=mode, model_dir=model_dir_encoder)
        elif(mode=='train' or mode=='predict'):
            model_dir_N = os.path.join(models_dir, 'ckpt_N')
            model_dir_C = os.path.join(models_dir, 'ckpt_C')
            self._utils.make_dir(model_dir_N)
            self._utils.make_dir(model_dir_C)

            self.trainer_N = self.initialize_trainer(mode=mode, model_dir=model_dir_N)
            self.trainer_C = self.initialize_trainer(mode=mode, model_dir=model_dir_C)

            self.Transformer_N = self.initialize_one_model(mode=mode, model_dir=model_dir_N)
            self.Transformer_C = self.initialize_one_model(mode=mode, model_dir=model_dir_C)


    def initialize_one_model(self, mode=None, model_dir=None) -> None:
        model_params = dict(configs=self._configs, meta=self._meta)
        loaded_model_params = dict(configs=self._configs, meta=self._meta)

        loaded_model=None
        if mode== 'pretrain':
            loaded_model = Transformer(pretrain_mode=True, **model_params)
        if mode== 'train':
            loaded_model = Transformer(**model_params)
        elif mode== 'predict':
            #ckpt_file = os.path.join(model_dir, 'checkpoints', 'best.ckpt')
            ckpt_files = glob.glob(os.path.join(model_dir, 'checkpoints', 'best*.ckpt'))
            ckpt_file = ckpt_files[0] if ckpt_files else None
            print('The .ckpt file loaded: '+ckpt_file)
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

        return loaded_model

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
        if self.Trainer_configs.get('accelerator') in ("cpu", "mps"):
            return "auto"
        elif self.Trainer_configs.get('devices') == 1:
            return "auto"
        elif torch.cuda.device_count() > 1:
            return DDPStrategy(find_unused_parameters=False, static_graph=True)
        else:
            return "auto"
