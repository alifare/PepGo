import sys
import numpy as np
import torch
from torch import nn

import math
import einops

from typing import Optional, Tuple, Union

from sortedcontainers import SortedDict, SortedSet
from torch.fx.experimental import normalize

from .utils import UTILS
from pprint import pprint
from .ModelMixin import ModelMixin

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


from .utils import UTILS

import torch.nn as nn
from typing import Sequence, List, Union, Iterable
#import dreams.utils.misc as utils
from math import ceil
#------------------------------------------------ For Peptide -------------------------------------------------------

class FloatEncoder(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            min_wavelength: float = 0.001,
            max_wavelength: float = 10000,
            learnable_wavelengths: bool = False,
    ) -> None:
        """Initialize the MassEncoder."""
        super().__init__()

        # Error checking:
        if min_wavelength <= 0:
            raise ValueError("'min_wavelength' must be greater than 0.")

        if max_wavelength <= 0:
            raise ValueError("'max_wavelength' must be greater than 0.")

        self.learnable_wavelengths = learnable_wavelengths

        # Get dimensions for equations:
        d_sin = math.ceil(d_model / 2)
        d_cos = d_model - d_sin

        base = min_wavelength / (2 * np.pi)
        scale = max_wavelength / min_wavelength
        sin_exp = torch.arange(0, d_sin).float() / (d_sin - 1)
        cos_exp = (torch.arange(d_sin, d_model).float() - d_sin) / (d_cos - 1)
        sin_term = base * (scale ** sin_exp)
        cos_term = base * (scale ** cos_exp)

        if not self.learnable_wavelengths:
            self.register_buffer("sin_term", sin_term)
            self.register_buffer("cos_term", cos_term)
        else:
            self.sin_term = torch.nn.Parameter(sin_term)
            self.cos_term = torch.nn.Parameter(cos_term)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        sin_mz = torch.sin(X[:, :, None] / self.sin_term)
        cos_mz = torch.cos(X[:, :, None] / self.cos_term)
        return torch.cat([sin_mz, cos_mz], axis=-1)

class PositionalEncoder(FloatEncoder):
    def __init__(
            self,
            d_model: int,
            min_wavelength: float = 1.0,
            max_wavelength: float = 1e5,
    ) -> None:
        """Initialize the MzEncoder."""
        super().__init__(
            d_model=d_model,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(X.shape[1]).type_as(self.sin_term)
        pos = einops.repeat(pos, "n -> b n", b=X.shape[0])
        sin_in = einops.repeat(pos, "b n -> b n f", f=len(self.sin_term))
        cos_in = einops.repeat(pos, "b n -> b n f", f=len(self.cos_term))

        sin_pos = torch.sin(sin_in / self.sin_term)
        cos_pos = torch.cos(cos_in / self.cos_term)
        encoded = torch.cat([sin_pos, cos_pos], axis=2)
        return encoded + X

class PeptideDecoder(ModelMixin):
    def __init__(
            self,
            n_tokens: int,
            d_model: int = 128,
            n_heads: int = 8,
            dim_feedforward: int = 1024,
            n_layers: int = 1,
            dropout: float = 0,
            positional_encoder: PositionalEncoder | bool = True,
            padding_int: int | None = None,
            max_charge: int = 10,
    ) -> None:
        """Initialize a PeptideDecoder."""
        super().__init__()
        self._d_model = d_model
        self._nhead = n_heads
        self._dim_feedforward = dim_feedforward
        self._n_layers = n_layers
        self._dropout = dropout

        if (isinstance(n_tokens, int) and isinstance(padding_int, int)):
            self._n_tokens = n_tokens
            self._padding_int = padding_int
        else:
            raise ValueError("n_tokens and padding_int must be specified as an int")

        if callable(positional_encoder):
            self.positional_encoder = positional_encoder
        elif positional_encoder:
            self.positional_encoder = PositionalEncoder(d_model)
        else:
            self.positional_encoder = torch.nn.Identity()

        self.token_encoder = torch.nn.Embedding(
            self._n_tokens + 1,
            d_model,
            padding_idx=self._padding_int,
        )

        # Additional model components
        layer = torch.nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_decoder = torch.nn.TransformerDecoder(
            layer,
            num_layers=n_layers,
        )

        self.charge_encoder = torch.nn.Embedding(max_charge, d_model)
        self.mass_encoder = FloatEncoder(d_model)

        self.final = torch.nn.Linear(
            d_model, self.token_encoder.num_embeddings
        )

    def embed(
            self,
            tokens: torch.Tensor | None,
            *args: torch.Tensor,
            memory: torch.Tensor | None,
            memory_key_padding_mask: torch.Tensor | None = None,
            memory_mask: torch.Tensor | None = None,
            tgt_mask: torch.Tensor | None = None,
            precursors: torch.Tensor | None = None,
            **kwargs,
    ) -> torch.Tensor:
        """Embed a collection of sequences.

        Parameters
        ----------
        tokens : list of str, torch.Tensor, or None
            The partial molecular sequences for which to predict the next
            token. Optionally, these may be the token indices instead
            of a string.
        *args : torch.Tensor, optional
            Additional data. These may be used by overwriting the
            `global_token_hook()` method in a subclass.
        memory : torch.Tensor of shape (batch_size, len_seq, d_model)
            The representations from a ``TransformerEncoder``, such as a
            ``SpectrumTransformerEncoder``.
        memory_key_padding_mask : torch.Tensor of shape (batch_size, len_seq)
            Passed to `torch.nn.TransformerEncoder.forward()`. The mask that
            indicates which elements of ``memory`` are padding.
        memory_mask : torch.Tensor
            Passed to `torch.nn.TransformerEncoder.forward()`. The mask
            for the memory sequence.
        tgt_mask : torch.Tensor or None
            Passed to `torch.nn.TransformerEncoder.forward()`. The default
            is a mask that is suitable for predicting the next element in
            the sequence.
        **kwargs : dict
            Additional data fields. These may be used by overwriting
            the `global_token_hook()` method in a subclass.

        Returns
        -------
        embeddings : torch.Tensor of size (batch_size, len_sequence, d_model)
            The output of the Transformer layer containing the embeddings
            of the tokens in the sequence. These may be tranformed to yield
            scores for token predictions using the `.score_embeddings()`
            method.

        """

        # Prepare sequences
        if tokens is None:
            tokens = torch.tensor([[]]).to(self.device)

        # Encode everything:
        encoded = self.token_encoder(tokens)

        # Add the global token
        masses = self.mass_encoder(precursors[:, None, 0]).squeeze(1)
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        global_token = masses + charges

        encoded = torch.cat([global_token[:, None, :], encoded], dim=1)

        # Create the padding mask:
        tgt_key_padding_mask = encoded.sum(axis=2) == 0
        tgt_key_padding_mask[:, 0] = False

        # Feed through model:
        encoded = self.positional_encoder(encoded)

        if tgt_mask is None:
            sz = encoded.shape[1]
            tgt_mask = ~torch.triu(torch.ones(sz, sz, dtype=torch.bool)).transpose(0, 1)
            tgt_mask = tgt_mask.to(self.device)

        return self.transformer_decoder(
            tgt=encoded,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_mask=memory_mask,
        )

    def forward(
            self,
            tokens: torch.Tensor | None,
            *args: torch.Tensor,
            memory: torch.Tensor | None,
            memory_key_padding_mask: torch.Tensor | None = None,
            memory_mask: torch.Tensor | None = None,
            tgt_mask: torch.Tensor | None = None,
            precursors: torch.Tensor | None = None,
            **kwargs,
    ) -> torch.Tensor:
        """Decode a collection of sequences.

        Parameters
        ----------
        tokens : list of str, torch.Tensor, or None
            The partial molecular sequences for which to predict the next
            token. Optionally, these may be the token indices instead
            of a string.
        *args : torch.Tensor, optional
            Additional data. These may be used by overwriting the
            `global_token_hook()` method in a subclass.
        memory : torch.Tensor of shape (batch_size, len_seq, d_model)
            The representations from a ``TransformerEncoder``, such as a
            ``SpectrumTransformerEncoder``.
        memory_key_padding_mask : torch.Tensor of shape (batch_size, len_seq)
            Passed to `torch.nn.TransformerEncoder.forward()`. The mask that
            indicates which elements of ``memory`` are padding.
        memory_mask : torch.Tensor
            Passed to `torch.nn.TransformerEncoder.forward()`. The mask
            for the memory sequence.
        tgt_mask : torch.Tensor or None
            Passed to `torch.nn.TransformerEncoder.forward()`. The default
            is a mask that is suitable for predicting the next element in
            the sequence.
        **kwargs : dict
            Additional data fields. These may be used by overwriting
            the `global_token_hook()` method in a subclass.

        Returns
        -------
        scores : torch.Tensor of size (batch_size, len_sequence, n_tokens)
            The raw output for the final linear layer. These can be Softmax
            transformed to yield the probability of each token for the
            prediction.

        """
        emb = self.embed(
            tokens,
            *args,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_mask=memory_mask,
            tgt_mask=tgt_mask,
            precursors=precursors,
            **kwargs,
        )
        return self.final(emb)

class PeptideTokenizer():
    def __init__(self, residues=None, replace_isoleucine_and_leucine_with_X=False, start_token=None, stop_token="$"):
        self.residues = residues.copy()
        self.start_token = start_token
        self.stop_token = stop_token
        self.replace_isoleucine_and_leucine_with_X = replace_isoleucine_and_leucine_with_X

        if self.replace_isoleucine_and_leucine_with_X:
            self.residues["X"] = self.residues["I"] if ("I" in self.residues) else self.residues["L"]
            if "I" in self.residues:
                del self.residues["I"]
            if "L" in self.residues:
                del self.residues["L"]

        tokens = SortedSet(self.residues)
        if self.stop_token in tokens:
            raise ValueError(
                f"Stop token {stop_token} already exists in tokens.",
            )

        if start_token is not None:
            tokens.add(self.start_token)
        if stop_token is not None:
            tokens.add(self.stop_token)

        self.index = SortedDict({k: i + 1 for i, k in enumerate(tokens)})
        self.reverse_index = [None] + list(tokens)  # 0 is padding.
        self.start_int = self.index.get(self.start_token, None)
        self.stop_int = self.index.get(self.stop_token, None)
        self.padding_int = 0

        self.extended_index = self.index.copy()
        self.extended_index[''] = 0

    def __len__(self) -> int:
        """The number of tokens."""
        return len(self.index)

    def split(self, sequence: str) -> list[str]:
        return (sequence.split(','))

    def tokenize(
            self,
            sequences: list,
            add_start: bool = False,
            add_stop: bool = False,
            to_strings: bool = False,
    ) -> torch.Tensor | list[list[str]]:

        add_start = add_start and self.start_token is not None
        add_stop = add_stop and self.stop_token is not None
        try:
            out = []
            for seq in sequences:
                tokens = seq
                if add_start and tokens[0] != self.start_token:
                    tokens.insert(0, self.start_token)

                if add_stop and tokens[-1] != self.stop_token:
                    tokens.append(self.stop_token)

                if to_strings:
                    out.append(tokens)
                    continue

                out.append(torch.tensor([self.extended_index[t] for t in tokens]))

            if to_strings:
                return out

            return nn.utils.rnn.pad_sequence(out, batch_first=True)
        except KeyError as err:
            raise ValueError("Unrecognized token") from err

    def detokenize(
            self,
            tokens: torch.Tensor,
            join: bool = True,
            trim_start_token: bool = True,
            trim_stop_token: bool = True,
    ) -> list[str] | list[list[str]]:
        decoded = []
        for row in tokens:
            seq = []
            for idx in row:
                if self.reverse_index[idx] is None:
                    continue

                if trim_stop_token and idx == self.stop_int:
                    break

                seq.append(self.reverse_index[idx])

            if trim_start_token and seq[0] == self.start_token:
                seq.pop(0)

            if join:
                seq = ",".join(seq)

            decoded.append(seq)

        return decoded

    def detokenize_residue(self, idx):
        residue = self.reverse_index[idx]
        return (residue)

#------------------------------------------------ For Spectrum -------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, depth=None, act_last=True, act=nn.ReLU, bias=True, dropout=0):
        super().__init__()

        if isinstance(hidden_dim, int):
            assert depth is not None
            hidden_dim = [hidden_dim] * depth
        elif hidden_dim == 'interpolated':
            assert depth is not None
            hidden_dim = self.interpolate_interval(a=in_dim, b=out_dim, n=depth - 1, only_inter=True, rounded=True)
        elif isinstance(hidden_dim, Sequence):  # e.g. is List or Tuple
            depth = len(hidden_dim)
        else:
            raise ValueError

        self.ff = nn.ModuleList([])
        for l in range(depth):
            d1 = hidden_dim[l - 1] if l != 0 else in_dim
            d2 = hidden_dim[l] if l != depth - 1 else out_dim
            self.ff.append(nn.Linear(d1, d2, bias=bias))
            if l != depth - 1:
                self.ff.append(nn.Dropout(p=dropout))
            if l != depth - 1 or act_last:
                self.ff.append(act())
        self.ff = nn.Sequential(*self.ff)

    @staticmethod
    def interpolate_interval(a, b, n, only_inter=False, rounded=False):
        """
        :param a: Start point.
        :param b: End point.
        :param n: Num. of steps.
        :param only_inter: Does not return interval ends a and b.
        :param rounded:
        """
        x_min, x_max = min(a, b), max(a, b)
        res = [x_min + i * (x_max - x_min) / (n + 1) for i in
               range(1 if only_inter else 0, n + 1 if only_inter else n + 2)]
        if x_max == a:
            res.reverse()
        if rounded:
            res = [round(x) for x in res]
        return res

    def forward(self, x):
        return self.ff(x)

class FourierFeatures(nn.Module):
    def __init__(self, strategy, x_min, x_max, trainable=True, funcs='both', sigma=10, num_freqs=512, fourier_power=2.0):
        assert strategy in {'random', 'voronov_et_al', 'lin_float_int', 'voronov_power'}
        assert funcs in {'both', 'sin', 'cos'}
        assert x_min < 1

        super().__init__()
        self.funcs = funcs
        self.strategy = strategy
        self.trainable = trainable
        self.num_freqs = num_freqs
        self.fourier_power = fourier_power

        if strategy == 'random':
            self.b = torch.randn(num_freqs) * sigma
        if self.strategy == 'voronov_et_al':
            self.b = torch.tensor(
                [1 / (x_min * (x_max / x_min) ** (2 * i / (num_freqs - 2))) for i in range(1, num_freqs)],
            )
        elif self.strategy == 'lin_float_int':
            self.b = torch.tensor(
                [1 / (x_min * i) for i in range(2, ceil(1 / x_min), 2)] +
                [1 / (1 * i) for i in range(2, ceil(x_max), 1)],
            )
        elif self.strategy == 'voronov_power':
            # ============================================================================
            # voronov_power strategy
            # Reference: Voronov et al., "Multi-scale Sinusoidal Embeddings Enable Learning on
            # High Resolution Mass Spectrometry Data", arXiv 2022.
            # DOI: https://doi.org/10.48550/arXiv.2207.02980
            # This is a modified version of the multi-scale sinusoidal embedding, with an
            # additional `power` parameter to control frequency distribution:
            #   t = (i / (N-1)) ** power
            #   freq = (1/x_min) * (x_max/x_min)^(-t)
            # where N = num_freqs, i = 0..N-1.
            #
            # - power = 1.0: uniform distribution on log scale (standard multi-scale).
            # - power > 1.0: more frequencies assigned to the high-frequency end
            #   (sensitive to small mass differences, e.g., < 1 Da).
            # - power < 1.0: more frequencies assigned to the low-frequency end
            #   (sensitive to large mass differences).
            #
            # For peptide sequencing, power > 1 is recommended to focus on fine mass
            # differences (e.g., K vs Q, 0.036 Da).
            # ============================================================================
            # 生成 0 到 N-1 的索引
            indices = torch.arange(num_freqs, dtype=torch.float32)
            # 归一化到 [0, 1]，并施加 power 次方
            t = (indices / (num_freqs - 1)) ** fourier_power
            # 计算频率：从低到高，但偏向高频
            # 注意：这里 1/x_min 是最大频率，1/x_max 是最小频率
            freqs = (1 / x_min) * ((x_max / x_min) ** (-t))
            self.b = freqs
        #pprint(self.b)
        self.b = self.b.unsqueeze(0)

        self.b = nn.Parameter(self.b, requires_grad=self.trainable)
        self.register_parameter('Fourier frequencies', self.b)

    def forward(self, x):
        x = 2 * torch.pi * x @ self.b
        if self.funcs == 'both':
            x = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
        elif self.funcs == 'cos':
            x = torch.cos(x)
        elif self.funcs == 'sin':
            x = torch.sin(x)
        return x

    def num_features(self):
        return self.b.shape[1] if self.funcs != 'both' else 2 * self.b.shape[1]

class MultiheadAttention(nn.Module):
    """
    MultiheadAttention module
    I learned a lot from https://github.com/pytorch/fairseq/blob/master/fairseq/modules/multihead_attention.py
    """
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.dropout = args.att_dropout
        self.use_transformer_bias = not args.no_transformer_bias
        self.attn_mech = args.attn_mech
        self.d_graphormer_params = args.d_graphormer_params

        if self.d_model % self.n_heads != 0:
            raise ValueError('Required: d_model % n_heads == 0.')

        self.head_dim = self.d_model // self.n_heads
        self.scale = self.head_dim ** -0.5

        # Parameters for linear projections of queries, keys, values and output
        self.weights = Parameter(torch.Tensor(4 * self.d_model, self.d_model))
        if self.use_transformer_bias:
            self.biases = Parameter(torch.Tensor(4 * self.d_model))

        if self.d_graphormer_params:
            self.lin_graphormer = nn.Linear(self.d_graphormer_params, self.n_heads, bias=False)

        # initializing
        # If we do Xavier normal initialization, std = sqrt(2/(2D))
        # but it's too big and causes unstability in PostNorm
        # so we use the smaller std of feedforward module, i.e. sqrt(2/(5D))
        mean = 0
        std = (2 / (5 * self.d_model)) ** 0.5
        nn.init.normal_(self.weights, mean=mean, std=std)
        if self.use_transformer_bias:
            nn.init.constant_(self.biases, 0.)

        if self.attn_mech == 'additive_v':
            self.additive_v = Parameter(torch.Tensor(self.n_heads, self.head_dim))
            nn.init.normal_(self.additive_v, mean=mean, std=std)

    def forward(self, q, k, v, mask, graphormer_dists=None, do_proj_qkv=True):
        """
        TODO: document shapes.

        :param q:
        :param k:
        :param v:
        :param mask:
        :param do_proj_qkv:
        :return:
        """
        bs, n, d = q.size()

        def _split_heads(tensor):
            bsz, length, d_model = tensor.size()
            return tensor.reshape(bsz, length, self.n_heads, self.head_dim).transpose(1, 2)

        if do_proj_qkv:
            q, k, v = self.proj_qkv(q, k, v)

        q = _split_heads(q)
        k = _split_heads(k)
        v = _split_heads(v)

        if self.attn_mech == 'dot-product':
            att_weights = torch.einsum('bhnd,bhdm->bhnm', q, k.transpose(-2, -1))
        elif self.attn_mech == 'additive_v' or self.attn_mech == 'additive_fixed':
            att_weights = (q.unsqueeze(-2) - k.unsqueeze(-3))
            if self.attn_mech == 'additive_v':
                att_weights = (att_weights * self.additive_v.unsqueeze(0).unsqueeze(2).unsqueeze(3))
            att_weights = att_weights.sum(dim=-1)
        else:
            raise NotImplementedError(f'"{self.attn_mech}" attention mechanism is not implemented.')
        att_weights = att_weights * self.scale

        if graphormer_dists is not None:
            if self.d_graphormer_params:
                # (bs, n, n, dists_d) -> (bs, n, n, n_heads) -> (bs, n_heads, n, n) = A.shape
                att_bias = self.lin_graphormer(graphormer_dists).permute(0, 3, 1, 2)
            else:
                # (bs, n, n, dists_d) -> (bs, 1, n, n) broadcastable with A
                att_bias = graphormer_dists.sum(dim=-1).unsqueeze(1)
            att_weights = att_weights + att_bias

        if mask is not None:
            att_weights.masked_fill_(mask.unsqueeze(1).unsqueeze(-1), -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = F.dropout(att_weights, p=self.dropout, training=self.training)
        _att_weights = att_weights.reshape(-1, n, n)
        output = torch.bmm(_att_weights, v.reshape(bs * self.n_heads, -1, self.head_dim))
        output = output.reshape(bs, self.n_heads, n, self.head_dim).transpose(1, 2).reshape(bs, n, -1)
        output = self.proj_o(output)

        return output, att_weights

    def proj_qkv(self, q, k, v):
        qkv_same = q.data_ptr() == k.data_ptr() == v.data_ptr()
        kv_same = k.data_ptr() == v.data_ptr()

        if qkv_same:
            q, k, v = self._proj(q, end=3 * self.d_model).chunk(3, dim=-1)
        elif kv_same:
            q = self._proj(q, end=self.d_model)
            k, v = self._proj(k, start=self.d_model, end=3 * self.d_model).chunk(2, dim=-1)
        else:
            q = self.proj_q(q)
            k = self.proj_k(k)
            v = self.proj_v(v)

        return q, k, v

    def _proj(self, x, start=0, end=None):
        weight = self.weights[start:end, :]
        bias = None if not self.use_transformer_bias else self.biases[start:end]
        return F.linear(x, weight=weight, bias=bias)

    def proj_q(self, q):
        return self._proj(q, end=self.d_model)

    def proj_k(self, k):
        return self._proj(k, start=self.d_model, end=2 * self.d_model)

    def proj_v(self, v):
        return self._proj(v, start=2 * self.d_model, end=3 * self.d_model)

    def proj_o(self, x):
        return self._proj(x, start=3 * self.d_model)

class tFeedForward(nn.Module):
    """FeedForward"""
    def __init__(self, args):
        super().__init__()
        self.dropout = args.ff_dropout
        self.d_model = args.d_model
        self.ff_dim = 4 * args.d_model
        self.use_transformer_bias = not args.no_transformer_bias

        self.in_proj = nn.Linear(self.d_model, self.ff_dim, bias=self.use_transformer_bias)
        self.out_proj = nn.Linear(self.ff_dim, self.d_model, bias=self.use_transformer_bias)

        # initializing
        mean = 0
        std = (2 / (self.ff_dim + self.d_model)) ** 0.5
        nn.init.normal_(self.in_proj.weight, mean=mean, std=std)
        nn.init.normal_(self.out_proj.weight, mean=mean, std=std)
        if self.use_transformer_bias:
            nn.init.constant_(self.in_proj.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x):
        # my preliminary experiments show all RELU-variants
        # work the same and slower, RELU FTW!!!
        y = F.relu(self.in_proj(x))
        y = F.dropout(y, p=self.dropout, training=self.training)
        return self.out_proj(y)

class ScaleNorm(nn.Module):
    """ScaleNorm"""
    def __init__(self, scale, eps=1e-5):
        super().__init__()
        self.scale = Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm

class TransformerEncoder(nn.Module):
    """Self-attention Transformer Encoder"""
    def __init__(self, args):
        super().__init__()
        self.residual_dropout = args.residual_dropout
        self.n_layers = args.n_layers
        self.pre_norm = args.pre_norm
        self._gradient_checkpointing = False

        self.atts = nn.ModuleList([MultiheadAttention(args) for _ in range(self.n_layers)])
        self.ffs = nn.ModuleList([tFeedForward(args) for _ in range(self.n_layers)])

        num_scales = self.n_layers * 2 + 1 if self.pre_norm else self.n_layers * 2
        if args.scnorm:
            self.scales = nn.ModuleList([ScaleNorm(args.d_model ** 0.5) for _ in range(num_scales)])
        else:
            self.scales = nn.ModuleList([nn.LayerNorm(args.d_model) for _ in range(num_scales)])

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def _layer_forward(self, i, x, src_mask, graphormer_dists):
        pre_norm = self.pre_norm
        post_norm = not pre_norm
        att = self.atts[i]
        ff = self.ffs[i]
        att_scale = self.scales[2 * i]
        ff_scale = self.scales[2 * i + 1]

        residual = x
        x = att_scale(x) if pre_norm else x
        x, _ = att(q=x, k=x, v=x, mask=src_mask, graphormer_dists=graphormer_dists)
        x = residual + F.dropout(x, p=self.residual_dropout, training=self.training)
        x = att_scale(x) if post_norm else x

        residual = x
        x = ff_scale(x) if pre_norm else x
        x = ff(x)
        x = residual + F.dropout(x, p=self.residual_dropout, training=self.training)
        x = ff_scale(x) if post_norm else x
        return x

    def forward(self, src_inputs, src_mask, graphormer_dists=None):
        x = F.dropout(src_inputs, p=self.residual_dropout, training=self.training)
        for i in range(self.n_layers):
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    self._layer_forward, i, x, src_mask, graphormer_dists,
                    use_reentrant=False
                )
            else:
                x = self._layer_forward(i, x, src_mask, graphormer_dists)

        x = self.scales[-1](x) if self.pre_norm else x
        return x

class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha=None, binary=False, return_softmax_out=False):
        """
        https://arxiv.org/pdf/1708.02002v2.pdf
        :param alpha: A vector summing up to one for multi-class classification, a positive-class scalar from (0, 1)
                      range for binary classification.
        :param return_softmax_out: If True, the return value of forward method is `(loss, softmax probabilities)` instead
               of `loss`.
        """
        super().__init__()
        self.binary = binary
        self.gamma = gamma
        assert gamma >= 0
        self.alpha = alpha
        if self.alpha is not None:
            if not self.binary:
                raise NotImplementedError('Alpha weighting is currently implemented only for binary classification.')
            assert (isinstance(alpha, float) and 0 < alpha < 1) or (isinstance(alpha, torch.Tensor) and alpha.sum() == 1)
        self.return_softmax_out = return_softmax_out

    def forward(self, inputs, targets):
        """
        :param inputs: Class logits of shape (..., num_classes).
        :param targets: One-hot class labels (..., num_classes).
        :return: Unreduced focal loss of shape (...).
        """

        if not self.binary:

            # Compute cross-entropy
            p = F.softmax(inputs, dim=-1)
            loss = F.nll_loss(p.log(), torch.argmax(targets, dim=-1), reduction='none')
            if self.gamma == 0:
                if self.return_softmax_out:
                    return loss, p
                return loss

            # Weight with focal loss terms
            p_t = (targets * p).sum(dim=-1)
            fl_term = (1 - p_t) ** self.gamma
            loss = fl_term * loss

            if self.return_softmax_out:
                return loss, p
            return loss

        else:
            weight = torch.ones(targets.shape, dtype=targets.dtype, device=targets.device)
            targets_mask = targets > 0

            if self.alpha is not None:
                weight[targets_mask] = self.alpha
                weight[~targets_mask] = 1 - self.alpha

            if self.gamma > 0:
                weight[targets_mask] *= (1 - inputs[targets_mask]) ** self.gamma
                weight[~targets_mask] *= inputs[~targets_mask] ** self.gamma

            return F.binary_cross_entropy(inputs, targets, weight=weight.detach())

class PeakEncoder(ModelMixin):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # ========== 1. 峰嵌入层 ==========
        self.ff_peak = FeedForward(
            in_dim=args.token_dim,
            out_dim=args.d_peak,
            hidden_dim=args.d_peak,
            depth=args.ff_peak_depth,
            dropout=args.dropout,
            bias=not args.no_ffs_bias
        )

        # ========== 2. 傅里叶特征编码器 ==========
        self.fourier_enc = FourierFeatures(
            strategy=args.fourier_strategy,
            num_freqs=args.fourier_num_freqs,
            fourier_power = args.fourier_power,
            x_min=args.min_mz, #x_min=args.dformat.max_tbxic_stdev if not args.fourier_min_freq else args.fourier_min_freq,
            x_max=args.max_mz, #x_max=args.dformat.max_mz,
            trainable=args.fourier_trainable
        )

        # 论文：4 层 FFN 将傅里叶特征映射到 d_fourier
        self.ff_fourier = FeedForward(
            in_dim=self.fourier_enc.num_features(),  # num_freqs * 2 (sin + cos)
            out_dim=args.d_fourier,
            hidden_dim=args.ff_fourier_d,
            depth=args.ff_fourier_depth,
            dropout=args.dropout,
            bias=not args.no_ffs_bias
        )

    def __normalize_spec(self, spectra):
        """
        只归一化 m/z 值，intensity 和 charge 保持不变
        """
        max_mz_in_data = spectra[..., 0].max().item()
        if max_mz_in_data > self.args.max_mz:
            raise ValueError('Max m/z value('+str(max_mz_in_data)+') in sepctra exceeded max_mz!')
        normalized_spec = spectra.clone()
        normalized_spec[..., 0] = normalized_spec[..., 0] / self.args.max_mz
        return normalized_spec

    def forward(self,
            spectra: torch.Tensor,
            charges: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        padding_mask = spectra[:, :, 0] == 0
        #self._utils.parse_var(padding_mask)

        #添加电荷特征（可选）
        if self.args.use_charge_feature:
            if charges is None:
                raise ValueError("charges must be provided when charge_feature=True")
            charge_features = (~padding_mask) * charges
            #self._utils.parse_var(charge_features)
            spectra = torch.cat([spectra, charge_features.unsqueeze(-1)], dim=-1)
        #self._utils.parse_var(spectra)

        peak_embs = self.ff_peak(self.__normalize_spec(spectra))
        #peak_embs = self.ff_peak(spectra)
        #self._utils.parse_var(peak_embs)

        # 傅里叶特征
        fourier_features = self.ff_fourier(self.fourier_enc(spectra[..., [0]]))
        #self._utils.parse_var(fourier_features)
        #拼接
        x = torch.cat([peak_embs, fourier_features], dim=-1)
        #self._utils.parse_var(x)

        #Graphormer 距离偏置
        graphormer_dists = None
        if self.args.graphormer_mz_diffs:
            graphormer_dists = fourier_features.unsqueeze(2) - fourier_features.unsqueeze(1)

        return x, padding_mask, graphormer_dists

class PeakDecoder(ModelMixin):
    def __init__(self, args):
        super().__init__()
        self._max_mz = args.max_mz
        self._bin_size = args.hot_mz_bin_size
        self._num_classes = self.num_hot_classes(max_val=self._max_mz, bin_size=self._bin_size)

        # ========== 预训练输出头 ==========
        # m/z 预测头（分类）
        self.ff_out = FeedForward(
            in_dim=args.d_model,
            hidden_dim=args.d_model,
            out_dim=self._num_classes,
            depth=args.ff_out_depth,
            act_last=False,
            dropout=args.dropout,
            bias=True
        )

        # 强度预测头（分类）
        self.ff_out_intens = FeedForward(
            in_dim=args.d_model,
            hidden_dim=args.d_model,
            out_dim=self.num_hot_classes(max_val=1.0, bin_size=0.05),  # 20 个强度 bin (0-1, 0.05步长)
            depth=args.ff_out_depth,
            act_last=False,
            dropout=args.dropout,
            bias=False
        )

    def num_hot_classes(self, max_val: float, bin_size: float) -> int:
        num_classes = max_val / bin_size
        assert num_classes == int(num_classes)
        return int(num_classes)

    def to_classes(self,
            vals: torch.Tensor,
            special_vals: List[float] = ()
    ) -> torch.Tensor:
        """ Assumes that last dimension of mzs is singleton. """
        special_masks = [vals == v for v in special_vals]
        classes = torch.round(vals / self._bin_size).long()
        classes = classes.clamp(max=self._num_classes - 1)  # clamp not to have a separate class for max_mz
        for i, m in enumerate(special_masks):
            classes[m] = self._num_classes + i
        return classes

    def to_hot(self, vals: torch.Tensor, dtype=torch.double):
        """ Assumes that last dimension of mzs is singleton. """
        classes = self.to_classes(vals)
        hots = F.one_hot(classes, num_classes=self._num_classes)
        return hots.squeeze(dim=-2).to(dtype)

    def forward(self, x):
        return self.ff_out(x)

class SpectrumEncoder(ModelMixin):
    def __init__(self, local_configs):
        super().__init__()
        class Args:
            pass

        args = Args()
        args.d_peak = local_configs.get('d_peak', 44)
        args.d_fourier = local_configs.get('d_fourier', 980)
        args.d_model = args.d_peak + args.d_fourier  # 1024
        args.n_layers = local_configs.get('n_layers', 7)
        args.n_heads = local_configs.get('n_heads', 8)
        args.ff_peak_depth = local_configs.get('ff_peak_depth', 1)
        args.dropout = local_configs.get('dropout', 0.1)
        args.min_mz = local_configs.get('min_mz', 0.0001)
        args.max_mz = local_configs.get('max_mz', 6000.0)
        args.hot_mz_bin_size = local_configs.get('hot_mz_bin_size', 0.05)
        args.fourier_num_freqs = local_configs.get('fourier_num_freqs', 1024)
        args.fourier_strategy = local_configs.get('fourier_strategy', 'lin_float_int')
        args.fourier_power = local_configs.get('fourier_power', 0.5)
        args.fourier_trainable = local_configs.get('fourier_trainable', False)
        args.ff_fourier_depth = local_configs.get('ff_fourier_depth', 5)
        args.ff_fourier_d = local_configs.get('ff_fourier_d', 512)
        args.ff_out_depth = local_configs.get('ff_out_depth', 1)
        args.graphormer_mz_diffs = local_configs.get('graphormer_mz_diffs', True)
        args.use_charge_feature = local_configs.get('use_charge_feature', False)
        args.pre_norm = local_configs.get('pre_norm', True)
        args.bias = local_configs.get('bias', False)
        args.no_ffs_bias = local_configs.get('no_ffs_bias', False)
        args.focal_loss_gamma = local_configs.get('focal_loss_gamma', 5)

        # 计算 token 维度
        args.token_dim = 2  # [m/z, intensity]
        if args.use_charge_feature:
            args.token_dim += 1  # [m/z, intensity, charge]

        args.att_dropout = args.dropout
        args.residual_dropout = args.dropout
        args.ff_dropout = args.dropout

        args.vanilla_transformer = False
        args.scnorm = False
        args.attn_mech = 'dot-product'
        args.no_transformer_bias = True  # 论文：无偏置
        args.d_graphormer_params = 0  # 论文中无参数，直接求和

        self._utils = UTILS()
        self.peak_encoder = PeakEncoder(args)
        self.transformer_encoder = TransformerEncoder(args)
        self.peak_decoder = PeakDecoder(args)

        # 损失函数
        self.mz_masking_loss = FocalLoss(gamma=args.focal_loss_gamma, return_softmax_out=True)

    def forward(
            self,
            spectra: torch.Tensor,
            charges: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        核心编码方法，返回所有 token 的完整嵌入（内部使用）

        Args:
            spectra: 谱图张量 (batch, 1+peaks, 2)，已包含 precursor token（强度 1.1）
            charges: 电荷数 (batch,1)，可选

        Returns:
            x: (batch, 1+peaks, d_model) 所有 token 的嵌入
                - x[:, 0, :] 是 precursor token 的嵌入（代表整个谱图）
                - x[:, 1:, :] 是碎片峰的嵌入
            padding_mask: (batch, 1+peaks) padding mask，True 表示填充位置
            graphormer_dists: Graphormer 距离偏置
        """
        x, padding_mask, graphormer_dists  = self.peak_encoder(spectra, charges)
        #self._utils.parse_var(graphormer_dists)
        #Transformer 编码
        x = self.transformer_encoder(x, padding_mask, graphormer_dists)

        #masked_embs
        return x, padding_mask, graphormer_dists