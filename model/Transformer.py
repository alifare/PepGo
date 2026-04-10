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

from .NeuralNetworks import PeakEncoder, SpectrumEncoder, PeptideDecoder, PeptideTokenizer

class Transformer(pl.LightningModule):
    def __init__(self, configs=None, meta=None, pretrain_mode: bool = False,  **kwargs: Dict):
        super().__init__()
        self.save_hyperparameters()

        self._configs = configs
        self._meta = meta
        self._proton = self._meta.proton
        self._mass_dict = self._meta.mass_dict
        self.residues = self._meta.tokens

        # 预训练专用 ===============================================
        self._pretrain_mode = pretrain_mode
        # 预训练专用END ===============================================

        self._utils = UTILS()


        Model_configs = self._configs.get('Model',{})
        Peptide_configs = Model_configs.get('Peptide',{})

        self.Transformer_configs = Model_configs.get('Transformer',{})

        self.Pretrain_configs = self._configs.get('Model', {}).get('Pretrain', {})
        self.Trainer_configs = self._configs.get('Model', {}).get('Trainer', {})


        MCTTS_configs = self._configs.get('MCTTS', {})
        MCTTS_Tree_configs = MCTTS_configs.get('Tree',{})
        MCTTS_Delta_configs = MCTTS_configs.get('Delta',{})

        self.max_peptide_len = Peptide_configs.get('max_peptide_len', 100)
        self.min_peptide_len = self.Transformer_configs.get('min_peptide_len', 6)

        self.replace_isoleucine_and_leucine_with_X = Peptide_configs.get('replace_isoleucine_and_leucine_with_X', False)
        self.isotope_error_range = tuple(MCTTS_Tree_configs.get('isotope_error_range', (0, 1)))

        n_log = self.Trainer_configs.get('n_log', 10)

        max_charge = self.Transformer_configs.get('max_charge', 10)
        dim_model = self.Transformer_configs.get('dim_model', 512)
        n_head = self.Transformer_configs.get('n_head', 8)
        dim_feedforward = self.Transformer_configs.get('dim_feedforward', 1024)
        n_layers = self.Transformer_configs.get('n_layers', 9)
        dropout = self.Transformer_configs.get('dropout',0.0)

        train_label_smoothing = self.Trainer_configs.get('train_label_smoothing', 0.01)

        # 预训练配置 ===============================================
        self.pretrain_temperature = self.Pretrain_configs.get('temperature', 0.07)
        self.pretrain_projection_dim = self.Pretrain_configs.get('projection_dim', 256)
        self.pretrain_augmentation_strength = self.Pretrain_configs.get('augmentation_strength', 0.05)
        self.pretrain_mask_ratio = self.Pretrain_configs.get('mask_ratio', 0.15)
        self.pretrain_mz_jitter_ratio = self.Pretrain_configs.get('mz_jitter_ratio',0.0001)
        self.use_masked_prediction = self.Pretrain_configs.get('use_masked_prediction', False)

        self.warmup_iters = None
        self.cosine_schedule_period_iters = None

        # 预训练配置END ===============================================

        self.tokenizer = PeptideTokenizer(
            residues=self._meta.tokens,
            replace_isoleucine_and_leucine_with_X=self.replace_isoleucine_and_leucine_with_X,
            start_token=None,
            stop_token="$"
        )

        self.vocab_size = len(self.tokenizer) + 1

        # Build the model.
        self.encoder = SpectrumEncoder(
            d_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
        )

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

        # ========== 预训练专用组件 ==========
        # 对比学习的投影头（仅在预训练模式使用）
        self.projection_head = nn.Sequential(
            nn.Linear(dim_model, self.pretrain_projection_dim),
            nn.ReLU(),
            nn.Linear(self.pretrain_projection_dim, self.pretrain_projection_dim)
        )

        # Masked spectrum prediction 头（可选）
        self.peak_predictor = nn.Linear(dim_model, 2)  # 预测 (mz, intensity)
        #========== 预训练专用组件END ==========

        self.softmax = torch.nn.Softmax(2)
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

        # ===== 1. 参数可训练性配置 =====
        if self._pretrain_mode:
            for param in self.decoder.parameters():
                param.requires_grad = False
            logger.info("Pretrain mode: Encoder + projection head trainable")
        else:
            freeze_encoder = self.Transformer_configs.get('freeze_encoder', False)
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                logger.info("Finetune mode: Encoder frozen")
            else:
                logger.info("Finetune mode: Full training")

    def configure_optimizers(self):
        """使用临时默认值创建优化器和调度器"""

        # 创建优化器
        if self._pretrain_mode:
            optimizer = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.projection_head.parameters()),
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

        # 创建调度器（使用临时默认值）
        lr_scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_iters=self.warmup_iters,
            cosine_schedule_period_iters=self.cosine_schedule_period_iters
        )

        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}

    # ==================== 数据处理方法 ====================
    def _process_batch(self, batch, type='labeled'):
        spectra, precursors, peptides = batch
        mzs = spectra[:, :, 0]
        intensities = spectra[:, :, 1]
        mzs = mzs.to(self.device)
        intensities = intensities.to(self.device)

        if(type=='labeled'):
            tokens = self.tokenizer.tokenize(peptides, add_stop=True)
            precursors = precursors.to(self.device)
            tokens = tokens.to(self.device)
            return(mzs, intensities, precursors, tokens)
        elif(type=='unlabeled'):
            return(mzs, intensities)
        else:
            raise ValueError('The mode must be labeled or unlabeled')

    # ==================== 数据增强方法 ====================
    def _augment_spectrum(self, mz: torch.Tensor, intensity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        质谱数据增强，创建正样本对

        Parameters
        ----------
        mz : torch.Tensor
            m/z values
        intensity : torch.Tensor
            intensity values

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Augmented mz and intensity tensors
        """
        # 添加强度噪声
        strength = self.pretrain_augmentation_strength
        intensity_aug = intensity * (1 + strength * torch.randn_like(intensity))

        # 随机丢弃部分峰 (dropout)
        mask = torch.rand_like(intensity) > self.pretrain_mask_ratio
        intensity_aug = intensity_aug * mask
        intensity_aug = torch.clamp(intensity_aug, min=0) # 确保强度非负（数值稳定性）

        # 可选：添加 m/z 小扰动
        mz_aug = mz + torch.randn_like(mz) * self.pretrain_mz_jitter_ratio * mz
        mz_aug = torch.clamp(mz_aug, min=0) #限制最小值,否则可能产生负m/z

        return mz_aug, intensity_aug

    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE contrastive loss

        Parameters
        ----------
        z1, z2 : torch.Tensor
            Projected representations of augmented views

        Returns
        -------
        torch.Tensor
            Contrastive loss
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # 计算相似度矩阵
        logits = torch.matmul(z1, z2.T) / self.pretrain_temperature
        labels = torch.arange(z1.shape[0]).to(z1.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def _masked_spectrum_loss(self, spectra: torch.Tensor, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Masked spectrum prediction loss

        Parameters
        ----------
        spectra : torch.Tensor
            Original spectra (mz, intensity)
        encoded : torch.Tensor
            Encoded representations
        mask : torch.Tensor
            Mask indicating which peaks are masked

        Returns
        -------
        torch.Tensor
            Reconstruction loss for masked peaks
        """
        # 预测被mask的峰
        pred = self.peak_predictor(encoded[:, 1:, :])  # 跳过全局token

        # 只计算mask位置的损失
        if mask.any():
            loss = F.mse_loss(pred[mask], spectra[mask])
        else:
            loss = torch.tensor(0.0, device=spectra.device)

        return loss

    # ==================== 前向传播 ====================
    def forward(self, batch):
        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' started '+ '+'*100)
        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)

        def supervised_forward():
            """监督学习前向传播"""
            _, _, peptides = batch
            mzs, ints, precursors, _ = self._process_batch(batch)
            memories, mem_masks = self.encoder(mzs, ints)
            return memories, mem_masks, precursors, peptides

        def pretrain_forward():
            """预训练前向传播"""
            mzs, ints = self._process_batch(batch, type='unlabeled')
            memories, mem_masks = self.encoder(mzs, ints)
            return memories, mem_masks

        """根据模式选择前向传播"""
        if self._pretrain_mode:
            return pretrain_forward()
        else:
            return supervised_forward()

    # ==================== 预训练步骤 ====================
    def _pretraining_step(self, batch, mode: str = "pretrain") -> torch.Tensor:
        """
        预训练步骤 - 对比学习

        Parameters
        ----------
        batch : tuple or dict
            Unlabeled spectrum batch
        mode : str
            Logging mode

        Returns
        -------
        torch.Tensor
            Loss value
        """
        mzs, intensities = self._process_batch(batch, type='unlabeled')

        # 创建两个增强视图
        mz1, int1 = self._augment_spectrum(mzs, intensities)
        mz2, int2 = self._augment_spectrum(mzs, intensities)

        # 编码两个视图
        latent1, _ = self.encoder(mz1, int1)
        latent2, _ = self.encoder(mz2, int2)

        # 取全局token（第一个位置）作为光谱表示
        z1 = self.projection_head(latent1[:, 0, :])
        z2 = self.projection_head(latent2[:, 0, :])

        # 计算对比损失
        loss = self._contrastive_loss(z1, z2)

        # 日志记录
        self.log(
            f"{mode}_contrastive_loss",
            loss.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True
        )

        # 可选：添加 masked spectrum prediction 损失
        if self.Pretrain_configs.get('use_masked_prediction', False):
            # 随机mask部分峰
            batch_size, n_peaks = mzs.shape
            mask = torch.rand(batch_size, n_peaks) < self.pretrain_mask_ratio
            mask = mask.to(self.device)

            # 创建masked输入
            spectra = torch.stack([mzs, intensities], dim=2)
            masked_spectra = spectra.clone()
            masked_spectra[mask] = 0

            # 重新编码masked spectra
            mz_masked = masked_spectra[:, :, 0]
            int_masked = masked_spectra[:, :, 1]
            latent_masked, _ = self.encoder(mz_masked, int_masked)

            # 计算重建损失
            recon_loss = self._masked_spectrum_loss(spectra, latent_masked, mask)
            loss = loss + recon_loss

            self.log(f"{mode}_recon_loss", recon_loss.detach(), on_step=True, on_epoch=True)

        return loss

    '''
    # ==================== 监督学习步骤 ====================
    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """监督学习的单步前向"""
        mzs, ints, precursors, tokens = self._process_batch(batch)
        memories, mem_masks = self.encoder(mzs, ints)

        scores = self.decoder(
            tokens=tokens,
            memory=memories,
            memory_key_padding_mask=mem_masks,
            precursors=precursors,
        )
        return scores, tokens
    '''

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
        mzs, ints, precursors, tokens = self._process_batch(batch)
        memories, mem_masks = self.encoder(mzs, ints)

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
            batch_size=pred.shape[0],
            prog_bar=True
        )
        self.log(
            f"{mode}_accuracy",
            acc.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
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
            if "pretrain_contrastive_loss" in self.trainer.callback_metrics:
                pretrain_loss = self.trainer.callback_metrics["pretrain_contrastive_loss"].detach().item()
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
            if "pretrain_val_contrastive_loss" in callback_metrics:
                metrics = {
                    "step": self.trainer.global_step,
                    "valid_pretrain": callback_metrics["pretrain_val_contrastive_loss"].detach().item(),
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