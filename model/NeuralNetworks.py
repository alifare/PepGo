import numpy as np
import torch
from torch import nn

import math
import einops

from sortedcontainers import SortedDict, SortedSet

from .utils import UTILS
from pprint import pprint
from .NNBase import NNBase

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

class PeakEncoder(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            min_mz_wavelength: float = 0.001,
            max_mz_wavelength: float = 10000,
            min_intensity_wavelength: float = 1e-6,
            max_intensity_wavelength: float = 1,
            learnable_wavelengths: bool = False,
    ) -> None:
        """Initialize the MzEncoder."""
        super().__init__()
        self.d_model = d_model
        self.learnable_wavelengths = learnable_wavelengths

        self.mz_encoder = FloatEncoder(
            d_model=self.d_model,
            min_wavelength=min_mz_wavelength,
            max_wavelength=max_mz_wavelength,
            learnable_wavelengths=learnable_wavelengths,
        )

        self.int_encoder = FloatEncoder(
            d_model=self.d_model,
            min_wavelength=min_intensity_wavelength,
            max_wavelength=max_intensity_wavelength,
            learnable_wavelengths=learnable_wavelengths,
        )
        self.combiner = torch.nn.Linear(2 * d_model, d_model, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        encoded = torch.cat(
            [
                self.mz_encoder(X[:, :, 0]),
                self.int_encoder(X[:, :, 1]),
            ],
            dim=2,
        )

        return self.combiner(encoded)

class SpectrumEncoder(NNBase):
    def __init__(
            self,
            d_model: int = 128,
            n_head: int = 8,
            dim_feedforward: int = 1024,
            n_layers: int = 1,
            dropout: float = 0,
    ):
        super().__init__()
        self._d_model = d_model
        self._nhead = n_head
        self._dim_feedforward = dim_feedforward
        self._n_layers = n_layers
        self._dropout = dropout

        self.peak_encoder = PeakEncoder(d_model)
        # self.peak_encoder = torch.nn.Linear(2, d_model)

        # The Transformer layers:
        layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            batch_first=True,
            dropout=self.dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=self.n_layers,
        )

        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, d_model))

    def forward(
            self,
            mz_array: torch.Tensor,
            intensity_array: torch.Tensor,
            *args: torch.Tensor,
            mask: torch.Tensor | None = None,
            **kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed a batch of mass spectra.

        Parameters
        ----------
        mz_array : torch.Tensor of shape (n_spectra, n_peaks)
            The zero-padded m/z dimension for a batch of mass spectra.
        intensity_array : torch.Tensor of shape (n_spectra, n_peaks)
            The zero-padded intensity dimension for a batch of mass spctra.
        *args : torch.Tensor
            Additional data. These may be used by overwriting the
            `global_token_hook()` method in a subclass.
        mask : torch.Tensor
            Passed to `torch.nn.TransformerEncoder.forward()`. The mask
            for the sequence.
        **kwargs : dict
            Additional data fields. These may be used by overwriting
            the `global_token_hook()` method in a subclass.

        Returns
        -------
        latent : torch.Tensor of shape (n_spectra, n_peaks + 1, d_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.

        """
        spectra = torch.stack([mz_array, intensity_array], dim=2)

        # Create the padding mask:
        src_key_padding_mask = spectra.sum(dim=2) == 0
        global_token_mask = torch.tensor([[False]] * spectra.shape[0]).type_as(
            src_key_padding_mask
        )
        src_key_padding_mask = torch.cat(
            [global_token_mask, src_key_padding_mask], dim=1
        )

        # Encode the peaks
        peaks = self.peak_encoder(spectra)

        # Add the precursor information:
        # latent_spectra = self.latent_spectrum.squeeze(0).expand(mz_array.shape[0], -1)
        # peaks = torch.cat([latent_spectra[:, None, :], peaks], dim=1)

        latent_spectra = self.latent_spectrum.expand(mz_array.shape[0], -1, -1)
        peaks = torch.cat([latent_spectra, peaks], dim=1)

        out = self.transformer_encoder(
            peaks,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        return out, src_key_padding_mask

class PeptideDecoder(NNBase):
    def __init__(
            self,
            n_tokens: int,
            d_model: int = 128,
            n_head: int = 8,
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
        self._nhead = n_head
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
            nhead=n_head,
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