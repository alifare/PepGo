import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import logging
logger = logging.getLogger('PepGo')

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import lightning.pytorch as pl
from .utils import UTILS
from pprint import pprint

import inspect



import dreams.utils.spectra as su

class Transformer(pl.LightningModule):
    def __init__(self, configs=None, meta=None, pretrain_mode: bool = False, encoder=None, **kwargs: Dict):
        super().__init__()
        self.save_hyperparameters()

        self._configs = configs
        self._meta = meta
        self._proton = self._meta.proton
        self._mass_dict = self._meta.mass_dict
        self.residues = self._meta.tokens
        self._pretrain_mode = pretrain_mode
        self._encoder_type = encoder
        assert self._encoder_type in ('DreaMS', 'Casanovo', 'my'), \
            f"encoder_type must be 'DreaMS' or 'Casanovo', got {self._encoder_type}"
        print(f"The encoder type is: {self._encoder_type}")
        #sys.exit()

        self._utils = UTILS()
        #self._utils.parse_var(self._pretrain_mode, 'self._pretrain_mode')

        Model_configs = self._configs.get('Model',{})
        Peptide_configs = Model_configs.get('Peptide',{})

        self.Transformer_configs = Model_configs.get('Transformer',{})
        self.DreaMS_configs = Model_configs.get('DreaMS', {})
        self.my_configs = Model_configs.get('my', {})

        self.Pretrain_configs = self._configs.get('Model', {}).get('Pretrain', {})
        self.Trainer_configs = self._configs.get('Model', {}).get('Trainer', {})
        self.Basic_configs = self._configs.get('Model', {}).get('Basic', {})

        MCTTS_configs = self._configs.get('MCTTS', {})
        MCTTS_Tree_configs = MCTTS_configs.get('Tree',{})
        MCTTS_Delta_configs = MCTTS_configs.get('Delta',{})

        self.max_peptide_len = Peptide_configs.get('max_peptide_len', 100)
        self.min_peptide_len = self.Transformer_configs.get('min_peptide_len', 6)
        self.max_mz = self.Basic_configs.get('max_mz', 2500)
        self.bin_size = self.Basic_configs.get('bin_size', 0.05)

        self.replace_isoleucine_and_leucine_with_X = Peptide_configs.get('replace_isoleucine_and_leucine_with_X', False)
        self.isotope_error_range = tuple(MCTTS_Tree_configs.get('isotope_error_range', (0, 1)))

        n_log = self.Trainer_configs.get('n_log', 10)

        max_charge = self.Transformer_configs.get('max_charge', 10)
        dim_model = self.Transformer_configs.get('dim_model', 1024)
        n_head = self.Transformer_configs.get('n_head', 8)
        dim_feedforward = self.Transformer_configs.get('dim_feedforward', 1024)
        n_layers = self.Transformer_configs.get('n_layers', 9)
        dropout = self.Transformer_configs.get('dropout',0.0)

        train_label_smoothing = self.Trainer_configs.get('train_label_smoothing', 0.01)

        self.warmup_iters = None
        self.cosine_schedule_period_iters = None

        # Build the model.
        self.encoder = None
        if (self._encoder_type == 'Casanovo'):
            from .Casanovo import SpectrumEncoder, PeptideDecoder, PeptideTokenizer
            self.encoder = SpectrumEncoder(
                d_model=dim_model,
                n_head=n_head,
                dim_feedforward=dim_feedforward,
                n_layers=n_layers,
                dropout=dropout,
            )
        elif (self._encoder_type == 'DreaMS'):
            from .DreaMS import SpectrumEncoder, PeptideDecoder, PeptideTokenizer
            self.encoder = SpectrumEncoder(self.DreaMS_configs)
        elif (self._encoder_type == 'my'):
            from .my import SpectrumEncoder, PeptideDecoder, PeptideTokenizer
            self.encoder = SpectrumEncoder(configs=self.my_configs)

        self.tokenizer = PeptideTokenizer(
            residues=self._meta.tokens,
            replace_isoleucine_and_leucine_with_X=self.replace_isoleucine_and_leucine_with_X,
            start_token=None,
            stop_token="$"
        )

        self.vocab_size = len(self.tokenizer) + 1


        #print(dir(self.encoder))
        #print("Attributes:", vars(self.encoder))
        #print("Methods and props:", [m for m in dir(self.encoder) if not m.startswith('_')])

        self.decoder = PeptideDecoder(
            n_tokens=len(self.tokenizer),
            d_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            padding_int=self.tokenizer.padding_int,
            max_charge=max_charge
        )

        ignore_index = 0
        self.celoss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, label_smoothing=train_label_smoothing
        )
        self.val_celoss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        # Optimizer settings.

        self.stop_token = self.tokenizer.stop_int

        # Logging.
        self.n_log = n_log
        self._history = []

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def setup(self, stage: str = None):
        """
        Lightning 的 setup 方法，在训练开始前自动调用
        """
        if stage != 'fit' and stage is not None:
            return
        self.num_gpus = self.trainer.num_devices

        # ===== 1. 参数可训练性配置 =====
        if self._pretrain_mode:
            for param in self.decoder.parameters():
                param.requires_grad = False
            logger.info("Pretrain mode: Encoder trainable (masked autoencoding)")
        else:
            freeze_encoder = self.Transformer_configs.get('freeze_encoder', False)
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                logger.info("Finetune mode: Encoder frozen")
            else:
                logger.info("Finetune mode: Full training")

    def configure_optimizers(self):
        # 创建优化器
        if self._pretrain_mode:
            optimizer = torch.optim.Adam(
                #list(self.encoder.parameters()) + list(self.loss_balancer.parameters()),
                list(self.encoder.parameters()),
                lr=self.Pretrain_configs.get('learning_rate', 3e-4),
                weight_decay=self.Pretrain_configs.get('weight_decay', 1e-5)
            )
            self.warmup_iters = self.Pretrain_configs.get('warmup_iters')
            self.cosine_schedule_period_iters = self.Pretrain_configs.get('cosine_schedule_period_iters')
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.Trainer_configs.get('learning_rate', 5e-4),
                weight_decay=self.Trainer_configs.get('weight_decay', 1e-5)
            )
            self.warmup_iters = self.Trainer_configs.get('warmup_iters')
            self.cosine_schedule_period_iters = self.Trainer_configs.get('cosine_schedule_period_iters')

        # 创建调度器
        lr_scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_iters=self.warmup_iters,
            cosine_schedule_period_iters=self.cosine_schedule_period_iters
        )

        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}

    # ==================== 数据处理方法 ====================
    def _process_batch(self, batch, datatype='labeled'):
        spectra, precursors, peptides, charges, precursors_mass, precursors_charge, precursors_mz = batch
        device = next(self.parameters()).device
        spectra = spectra.to(device)
        charges = charges.to(device)
        precursors_mass = precursors_mass.to(device)
        precursors_charge = precursors_charge.to(device)
        precursors_mz = precursors_mz.to(device)

        '''
        print(charges == precursors_charge)
        self._utils.parse_var(charges)
        self._utils.parse_var(precursors_charge)
        sys.exit()
        '''

        if(datatype=='labeled'):
            precursors = precursors.to(device)
            tokens = self.tokenizer.tokenize(peptides, add_stop=True)
            tokens = tokens.to(device)
            return(spectra, precursors, tokens, charges, precursors_mass, precursors_charge, precursors_mz)
        elif(datatype=='unlabeled'):
            return(spectra, precursors_mass, precursors_charge, precursors_mz)
        else:
            raise ValueError('The datatype must be labeled or unlabeled')

    # ==================== 前向传播 ====================
    def forward(self, batch):
        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' started '+ '+'*100)
        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)

        def supervised_forward():
            """监督学习前向传播"""
            _, _, peptides, _ = batch
            spectra, precursors, tokens, charges, precursors_mass, precursors_charge, precursors_mz = self._process_batch(batch, datatype='labeled')
            memories, mem_masks, _ = self.encoder(spectra, precursors_mass, precursors_charge, precursors_mz)
            return memories, mem_masks, precursors, peptides

        def pretrain_forward():
            """预训练前向传播"""
            spectra, charges = self._process_batch(batch, datatype='unlabeled')
            spectrum_embedding, peak_embeddings, _ = self.encoder(spectra, charges)
            return spectrum_embedding, peak_embeddings

        """根据模式选择前向传播"""
        if self._pretrain_mode:
            return pretrain_forward()
        else:
            return supervised_forward()

    # ==================== 预训练步骤 ====================
    def generate_mask(self, spectra, mask_ratio=0.3, min_n_masks=2, mode='random', deterministic_seed=None):
        """
        生成掩码（与 DreaMS 官方保持一致）

        Parameters
        ----------
        spectra : torch.Tensor
            谱图张量 (batch, n_peaks, 2)，最后一维为 [m/z, intensity]
        mask_ratio : float
            掩码比例，默认 0.3（与论文一致）
        min_n_masks : int
            最小掩码数量，默认 2
        mode : str
            'random': 按强度概率采样（训练）
            'fixed':  掩码强度最高的峰（验证用）
            'uniform': 均匀随机采样（不按强度）
        deterministic_seed : float, optional
            若提供，则用该值作为随机种子（例如使用 precursor m/z），实现确定性掩码

        Returns
        -------
        spectra_masked : torch.Tensor
            掩码后的谱图 (batch, n_peaks, 2)，仅 m/z 列被掩码，intensity 不变
        mask : torch.Tensor
            布尔掩码 (batch, n_peaks)，True 表示被掩码的位置
        """
        batch_size, n_peaks, _ = spectra.shape
        device = spectra.device

        mzs = spectra[..., 0]
        intensities = spectra[..., 1]

        # 有效峰：非填充（intensity > 0）且非 precursor（intensity < 1.0，因为 precursor 强度为 1.1）
        valid_mask = (intensities > 0) & (intensities <= 1.0)  # DreaMS 中 precursor 强度为 1.1 不参与掩码

        mask = torch.zeros_like(mzs, dtype=torch.bool)

        for b in range(batch_size):
            valid_idx = valid_mask[b].nonzero(as_tuple=True)[0]
            if len(valid_idx) == 0:
                continue

            n_peaks_valid = len(valid_idx)
            n_masks = max(min_n_masks, round(n_peaks_valid * mask_ratio))
            n_masks = min(n_masks, n_peaks_valid)

            # 确定性种子（可选）
            if deterministic_seed is not None:
                seed = int(deterministic_seed[b] if isinstance(deterministic_seed,
                                                               (list, torch.Tensor)) else deterministic_seed)
                torch.manual_seed(seed)

            if mode == 'random':
                # 按强度加权采样
                probs = intensities[b][valid_idx]
                probs = probs / (probs.sum() + 1e-8)
                if n_masks == n_peaks_valid:
                    masked_idx = torch.randperm(n_peaks_valid, device=device)
                else:
                    masked_idx = torch.multinomial(probs, n_masks, replacement=False)
            elif mode == 'fixed':
                # 掩码强度最高的峰
                valid_intensities = intensities[b][valid_idx]
                _, masked_idx = torch.topk(valid_intensities, n_masks)
            elif mode == 'uniform':
                shuffled_idx = torch.randperm(len(valid_idx), device=device)
                masked_idx = shuffled_idx[:n_masks]
            else:
                raise ValueError(f"Unknown mask mode: {mode}")

            mask[b, valid_idx[masked_idx]] = True

        # 仅将 m/z 设为 -1.0，intensity 保持不变
        spectra_masked = spectra.clone()
        spectra_masked[..., 0] = torch.where(mask, torch.tensor(-1.0, device=device), spectra_masked[..., 0])
        # intensity 不变（不修改）

        return spectra_masked, mask

    def _pretraining_step(self, batch, mode: str = "pretrain") -> torch.Tensor:
        spectra, precursors_mass, precursors_charge, precursors_mz = self._process_batch(batch, datatype='unlabeled')
        current_batch_size = spectra.shape[0]
        mask_mode = 'random' if mode == 'pretrain' else 'fixed'

        masked_spectra, mask = self.generate_mask(spectra, mode=mask_mode)

        if (self._encoder_type == 'Casanovo'):
            memories, _, _ = self.encoder(masked_spectra, precursors_charge)
        elif (self._encoder_type == 'DreaMS'):
            pass
        elif (self._encoder_type == 'my'):
            memories, mem_masks, _ = self.encoder(masked_spectra, precursors_mass, precursors_charge, precursors_mz)

        masked_embs = memories[mask]
        real = spectra[mask]

        pred_mz = self.encoder.peak_decoder(masked_embs)
        real_mz = self.encoder.peak_decoder.to_hot(real[..., [0]])

        # 使用交叉熵（或自定义的 mz_masking_loss，但应保持一致）
        mz_loss = F.cross_entropy(pred_mz, real_mz, reduction='none')  # 或 self.encoder.mz_masking_loss(...)
        #mz_loss, _ = self.encoder.mz_masking_loss(pred_mz, real_mz)

        loss = mz_loss.mean()  # 无强度损失

        self.log(
            f"{mode}_mz_loss",
                loss.detach(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=current_batch_size,
                prog_bar=True
        )
        return loss

    def _supervised_step(self, batch, mode: str = "train") -> torch.Tensor:
        """
        监督学习步骤

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Labeled batch
        mode : str
            'train' or 'valid'

        Returns
        -------
        torch.Tensor
            Loss value
        """
        spectra, precursors, tokens, charges, precursors_mass, precursors_charge, precursors_mz = self._process_batch(batch, datatype='labeled')
        current_batch_size = spectra.shape[0]

        if (self._encoder_type == 'Casanovo'):
            memories, mem_masks, _ = self.encoder(spectra, precursors_charge)
        elif (self._encoder_type == 'DreaMS'):
            pass
        elif (self._encoder_type == 'my'):
            memories, mem_masks, _ = self.encoder(spectra, precursors_mass, precursors_charge, precursors_mz)

        scores = self.decoder(
            tokens=tokens,
            memory=memories,
            memory_key_padding_mask=mem_masks,
            precursors=precursors,
        )

        #pred, truth = self._forward_step(batch)
        pred = scores
        truth = tokens

        pred = pred[:, :-1, :].reshape(-1, self.vocab_size)

        if mode == "train":
            loss = self.celoss(pred, truth.flatten())
        else:
            loss = self.val_celoss(pred, truth.flatten())

        # 计算准确率
        pred_tokens = pred.argmax(dim=-1)
        acc = (pred_tokens == truth.flatten()).float().mean()

        self.log(
            f"{mode}_CELoss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=current_batch_size,
            prog_bar=True
        )
        self.log(
            f"{mode}_accuracy",
            acc.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=current_batch_size,
            prog_bar=True
        )
        return loss

    # ==================== Lightning 训练步骤 ====================
    def training_step(self, batch, *args) -> torch.Tensor:
        """
        统一的训练步骤，根据模式自动选择
        """
        if self._pretrain_mode:
            return self._pretraining_step(batch, mode="pretrain")
        else:
            return self._supervised_step(batch, mode="train")

    def validation_step(self, batch, *args) -> torch.Tensor:
        """
        统一的验证步骤
        """
        if self._pretrain_mode:
            return self._pretraining_step(batch, mode="pretrain_val")
        else:
            return self._supervised_step(batch, mode="valid")

    # ==================== 回调方法 ====================
    def on_train_epoch_end(self) -> None:
        """记录训练损失"""
        if not self._pretrain_mode:
            if "train_CELoss" in self.trainer.callback_metrics:
                train_loss = self.trainer.callback_metrics["train_CELoss"].detach().item()
            else:
                train_loss = np.nan
            metrics = {"step": self.trainer.global_step, "train": train_loss}
            self._history.append(metrics)
            self._log_history()
        else:
            # 预训练模式的日志
            if "pretrain_mz_loss" in self.trainer.callback_metrics:
                pretrain_loss = self.trainer.callback_metrics["pretrain_mz_loss"].detach().item()
                metrics = {"step": self.trainer.global_step, "pretrain": pretrain_loss}
                self._history.append(metrics)
                self._log_history()

    def on_validation_epoch_end(self) -> None:
        """记录验证指标"""
        callback_metrics = self.trainer.callback_metrics

        if not self._pretrain_mode:
            if "valid_CELoss" in callback_metrics:
                metrics = {
                    "step": self.trainer.global_step,
                    "valid": callback_metrics["valid_CELoss"].detach().item(),
                }
                self._history.append(metrics)
                self._log_history()
        else:
            if "pretrain_val_mz_loss" in callback_metrics:
                metrics = {
                    "step": self.trainer.global_step,
                    "valid_pretrain": callback_metrics["pretrain_val_mz_loss"].detach().item(),
                }
                self._history.append(metrics)
                self._log_history()

    def on_train_start(self):
        """记录优化器设置"""
        self.log(
            "hp/optimizer_warmup_iters",
            self.warmup_iters,
            on_epoch=True,
            sync_dist=True,
            reduce_fx="mean"
        )

        self.log(
            "hp/optimizer_cosine_schedule_period_iters",
            self.cosine_schedule_period_iters,
            on_epoch=True,
            sync_dist=True,
            reduce_fx="mean"
        )

    def _log_history(self) -> None:
        """输出日志"""
        if len(self._history) == 0:
            return
        if len(self._history) == 1:
            if not self._pretrain_mode:
                header = "Step\tTrain loss\tValid loss\t"
            else:
                header = "Step\tPretrain loss\t"
            logger.info(header)

        metrics = self._history[-1]
        if metrics["step"] % self.n_log == 0:
            if not self._pretrain_mode:
                msg = "%i\t%.6f\t%.6f"
                vals = [
                    metrics["step"],
                    metrics.get("train", np.nan),
                    metrics.get("valid", np.nan),
                ]
            else:
                msg = "%i\t%.6f"
                vals = [
                    metrics["step"],
                    metrics.get("pretrain", np.nan),
                ]
            logger.info(msg, *vals)

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm-up followed by cosine
    shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup_iters : int
        The number of iterations for the linear warm-up of the learning
        rate.
    cosine_schedule_period_iters : int
        The number of iterations for the cosine half period of the
        learning rate.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_iters: int, cosine_schedule_period_iters: int):
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    '''
    #OLD codes
    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (
                1 + np.cos(np.pi * epoch / self.cosine_schedule_period_iters)
        )
        if epoch <= self.warmup_iters:
            lr_factor *= epoch / self.warmup_iters
        return lr_factor
    '''

    def get_lr_factor(self, epoch):
        if self.cosine_schedule_period_iters <= 0:
            return 1.0

        progress = min(epoch, self.cosine_schedule_period_iters) / self.cosine_schedule_period_iters
        lr_factor = 0.5 * (1 + np.cos(np.pi * progress))

        if epoch <= self.warmup_iters:
            lr_factor *= epoch / max(self.warmup_iters, 1)

        return lr_factor
