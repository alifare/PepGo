import sys

import numpy as np
import pprint as pp
import torch
from torch import nn
from pygments.lexers.ruby import FancyLexer
from torch.utils.tensorboard import SummaryWriter

import einops
import heapq

import logging
logger = logging.getLogger('PepGo')

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from collections.abc import Callable
from depthcharge.encoders import FloatEncoder, PeakEncoder, PositionalEncoder
from depthcharge.tokenizers import Tokenizer

from depthcharge.transformers import (
    AnalyteTransformerDecoder,
    SpectrumTransformerEncoder,
)

#from depthcharge.tokenizers import PeptideTokenizer
import depthcharge

import lightning.pytorch as pl

from .utils import UTILS

class SpectrumEncoder(SpectrumTransformerEncoder):
    def __init__(
        self,
        d_model: int = 128,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        peak_encoder: PeakEncoder | Callable | bool = True,
    ):
        """Initialize a SpectrumEncoder."""
        super().__init__(d_model, n_head, dim_feedforward, n_layers, dropout, peak_encoder)

        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, d_model))

    def global_token_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        *args: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:

        return(self.latent_spectrum.squeeze(0).expand(mz_array.shape[0], -1))

class PeptideTokenizer(depthcharge.tokenizers.PeptideTokenizer):
    residues = {}

    #@abstractmethod
    def split(self, sequence: str) -> list[str]:
        return(sequence.split(','))

    def tokenize(
        self,
        sequences: list,
        add_start: bool = False,
        add_stop: bool = False,
        to_strings: bool = False,
    ) -> torch.Tensor | list[list[str]]:
        """Tokenize the input sequences.

        Parameters
        ----------
        sequences : Iterable[str] or str
            The sequences to tokenize.
        add_start : bool, optional
            Prepend the start token to the beginning of the sequence.
        add_stop : bool, optional
            Append the stop token to the end of the sequence.
        to_strings : bool, optional
            Return each as a list of token strings rather than a
            tensor. This is useful for debugging.

        Returns
        -------
        torch.tensor of shape (n_sequences, max_length) or list[list[str]]
            Either a tensor containing the integer values for each
            token, padded with 0's, or the list of tokens comprising
            each sequence.

        """
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

                out.append(torch.tensor([self.index[t] for t in tokens]))

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
        """Retreive sequences from tokens.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_sequences, max_length)
            The zero-padded tensor of integerized tokens to decode.
        join : bool, optional
            Join tokens into strings?
        trim_start_token : bool, optional
            Remove the start token from the beginning of a sequence.
        trim_stop_token : bool, optional
            Remove the stop token and anything following it from the sequence.

        Returns
        -------
        list[str] or list[list[str]]
            The decoded sequences each as a string or list or strings.

        """
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


class PeptideDecoder(AnalyteTransformerDecoder):
    def __init__(
        self,
        n_tokens: int | Tokenizer,
        d_model: int = 128,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        positional_encoder: PositionalEncoder | bool = True,
        padding_int: int | None = None,
        max_charge: int = 4,
    ) -> None:
        """Initialize a PeptideDecoder."""

        super().__init__(
            n_tokens=n_tokens,
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            positional_encoder=positional_encoder,
            padding_int=padding_int,
        )

        self.charge_encoder = torch.nn.Embedding(max_charge, d_model)
        self.mass_encoder = FloatEncoder(d_model)
        self._utils = UTILS()

        # Override the output layer to have +1 in the second dimension
        # compared to the AnalyteTransformerDecoder to account for
        # padding as a possible class (=0) and avoid problems during
        # beam search decoding.
        self.final = torch.nn.Linear(
            d_model, self.token_encoder.num_embeddings
        )

    def global_token_hook(
        self,
        tokens: torch.Tensor,
        precursors: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        Override global_token_hook to include precursor information.

        Parameters
        ----------
        *args :
        tokens : list of str, torch.Tensor, or None
            The partial molecular sequences for which to predict the
            next token. Optionally, these may be the token indices
            instead of a string.
        precursors : torch.Tensor
            Precursor information.
        *args : torch.Tensor
            Additional data passed with the batch.
        **kwargs : dict
            Additional data passed with the batch.

        Returns
        -------
        torch.Tensor of shape (batch_size, d_model)
            The global token representations.
        """
        masses = self.mass_encoder(precursors[:, None, 0]).squeeze(1)
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges
        return precursors


    def generate_tgt_mask(self, sz):
        return ~torch.triu(torch.ones(sz, sz, dtype=torch.bool)).transpose(0, 1)

    def myTokenize(self, sequence, partial=False):
        print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' started '+ '+'*100)

        self._utils.parse_var(sequence)
        if not isinstance(sequence[0], str):
            return sequence  # Assume it is already tokenized.

        #if self.reverse:
        #    sequence = list(reversed(sequence))

        if not partial:
            sequence += ["$"]

        #tokens = [self._aa2idx[aa] for aa in sequence]
        tokens = [self.index[aa] for aa in sequence]
        tokens = torch.tensor(tokens, device=self.device)

        print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)
        return(tokens)

    def tokenize_residue(self, residue):
        token = self._aa2idx[residue]
        return(token)

    def detokenize_residue(self, token):
        #residue = self._idx2aa.get(token.item())
        residue = self._idx2aa.get(token)
        return(residue)

    def forward(self, peptides, precursors, memory, memory_key_padding_mask, partial=False):
        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' started '+ '+'*100)
        if(peptides is not None):
            tokens = [self.myTokenize(s, partial) for s in peptides]
            tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        else:
            tokens = torch.tensor([[]]).to(self.device)

        masses = self.mass_encoder(precursors[:, None, 0])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]

        if(peptides is None):
            tgt = precursors
        else:
            tgt = torch.cat([precursors, self.aa_encoder(tokens)], dim=1)

        tgt_key_padding_mask = tgt.sum(axis=2) == 0
        tgt = self.pos_encoder(tgt)
        tgt_mask = self.generate_tgt_mask(tgt.shape[1]).to(self.device)

        preds = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask.to(self.device),
        )

        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)
        return self.final(preds), tokens


class Transformer(pl.LightningModule):
    def __init__(self,
        dim_model: int = 512,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 9,
        dropout: float = 0.0,
        dim_intensity: Optional[int] = None,
        max_peptide_len: int = 100,
        max_length: int = 100,
        residues: Union[Dict[str, float], str] = "canonical",
        max_charge: int = 5,
        precursor_mass_tol: float = 50,
        isotope_error_range: Tuple[int, int] = (0, 1),
        min_peptide_len: int = 6,
        n_beams: int = 1,
        top_match: int = 1,
        n_log: int = 10,
        tb_summarywriter: Optional[
            torch.utils.tensorboard.SummaryWriter
        ] = None,
        train_label_smoothing: float = 0.01,
        warmup_iters: int = 100_000,
        cosine_schedule_period_iters: int = 600_000,
        max_iters: int = 600_000,
        #out_writer: Optional[ms_io.MztabWriter] = None,
        out_writer = None,
        calculate_precision: bool = False,
        tokenizer=None,
        meta = None,
        **kwargs: Dict,
    ):
        super().__init__()
        self._meta = meta
        self._proton = self._meta.proton
        self._mass_dict = self._meta.mass_dict
        self._utils = UTILS()
        self.save_hyperparameters()

        '''
        print('self._meta.tokens',end=':')
        print(len(self._meta.tokens))
        pp.pprint(self._meta.tokens)
        '''

        self.tokenizer = tokenizer or PeptideTokenizer(residues=self._meta.tokens, start_token=None, stop_token="$")
        self.vocab_size = len(self.tokenizer) + 1

        #print('self.vocab_size', self.vocab_size)
        '''
        print('self.tokenizer.index',end=':')
        print(len(self.tokenizer.index))
        pp.pprint(dict(self.tokenizer.index))
        print('-'*100)
        '''

        '''
        print('self.tokenizer.reverse_index',end=':')
        print(len(self.tokenizer.reverse_index))
        print(type(self.tokenizer.reverse_index))
        for i in range(len(self.tokenizer.reverse_index)):
            print(i, self.tokenizer.reverse_index[i])
        print('-'*100)
        '''


        '''
        # 方法1: difference() 方法
        diff1 = set(self.tokenizer.reverse_index).difference(set(self.tokenizer.index))
        diff2 = set(self.tokenizer.index).difference(set(self.tokenizer.reverse_index))
        print('diff1-2',end=':')  # 输出: {1, 2, 3}
        print(len(diff1),len(diff2))
        print(diff1,diff2)
        '''


        '''
        print('PeptideDecoder',end=':')
        self._utils.parse_class(PeptideTokenizer)
        print('-'*100)
        '''


        '''
        print('-'*100)
        print('self.tokenizer:')
        self._utils.parse_instance(self.tokenizer)
        '''

        self.encoder = SpectrumEncoder(
            d_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.decoder = PeptideDecoder(
            n_tokens=self.tokenizer,
            d_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
        )

        #self._utils.parse_var(self.peptide_mass_calculator.masses.items())
        #self._utils.parse_var(residues)

        #Function parameters() from torch.nn.Module returns an iterator over module parameters.
        #This is typically passed to an optimizer.

        #print('self.parameters()',end=':')
        #print(next(self.parameters()).device)
        #for param in self.parameters():
            #print(type(param), param.size())
            #print(param, type(param), param.size())

    '''
    #def forward(self, peptides, precursors, memory, memory_key_padding_mask, partial=False):
    def forward(self, batch: Dict[str, torch.Tensor]) -> List[List[Tuple[float, np.ndarray, str]]]:
        print(self.__class__.__name__ + ' ' + sys._getframe().f_code.co_name + ' started ' + '+' * 100)
        #self._utils.parse_var(batch)
        mzs, intensities, precursors, seqs = self._process_batch(batch)
        #print(seqs)

        print('\nmzs:')
        print(mzs.shape)
        pp.pprint(mzs)

        print('\nintensities:')
        print(intensities.shape)
        pp.pprint(intensities)


        print('\nseqs:')
        #print(seqs.shape)
        pp.pprint(seqs)
        sys.exit()
        print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)
        return(True)


    def _get_top_peptide(
        self,
        pred_cache: Dict[int, List[Tuple[float, float, np.ndarray, torch.Tensor]]],
    ) -> Iterable[List[Tuple[float, np.ndarray, str]]]:

        for peptides in pred_cache.values():
            if len(peptides) > 0:
                yield [
                    (
                        pep_score,
                        aa_scores,
                        ",".join(self.decoder.detokenize(pred_tokens)),
                    )
                    for pep_score, _, aa_scores, pred_tokens in heapq.nlargest(
                        self.top_match, peptides
                    )
                ]
            else:
                yield []


    def validation_step(self, batch, *args) -> torch.Tensor:
        #print('')
        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' started '+ '+'*100)
        spectra, precursors, peptides = batch

        peptides_true = [','.join(i) for i in peptides]

        # Record the loss.
        loss = self.training_step(batch, mode="valid")
        if not self.calculate_precision:
            return loss

        # Calculate and log amino acid and peptide match evaluation metrics from the predicted peptides.
        peptides_pred = self.beam_pred(spectra, precursors)
        aa_precision, aa_recall, pep_precision = self.evaluate(peptides_true, peptides_pred)

        log_args = dict(on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "Peptide precision at coverage=1",
            pep_precision,
            **log_args,
        )
        self.log(
            "AA precision at coverage=1",
            aa_precision,
            **log_args,
        )

        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)
        return(loss)


    def predict_step(self, batch, *args):
        #print('\n'+self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' started '+ '+'*100)
        spectra, precursors, peptides = batch
        peptides_pred = self.beam_pred(spectra, precursors)

        outputs=[peptides_pred, peptides]

        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)
        return(outputs)

    def beam_pred(self, spectra, precursors):
        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' started '+ '+'*100)
        #spectra = torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True)
        #precursors = torch.tensor(precursors)

        peptides_pred = []
        for spectrum_preds in self.forward(spectra, precursors):
            if(not spectrum_preds):
                peptides_pred.append('')
                continue
            for _, _, pred in spectrum_preds:
                if(not pred):
                    pred=''
                peptides_pred.append(pred)

        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)
        return(peptides_pred)

    def on_predict_batch_end(self, outputs, *args) -> None:
        #print('\n'+self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' started '+ '+'*100)
        predictions, peptides = outputs

        for i in range(len(peptides)):
            print(str(predictions[i])+'\n'+','.join(peptides[i])+'\n')

        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)

    def evaluate(self, peptides_true, peptides_pred):
        #print('\n'+self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' started '+ '+'*100)
        if(len(peptides_true)  != len(peptides_pred)):
            sys.exit('peptides_true and peptides_pred must have the same size!')

        pep_eval_arr=[]
        for i in range(len(peptides_true)):
            pep_true = peptides_true[i]
            pep_pred = peptides_pred[i]
            pep_true_aa = pep_true.split(',')
            pep_pred_aa = pep_pred.split(',')
            pep_true_aa_len = len(pep_true_aa)
            pep_pred_aa_len = len(pep_pred_aa)

            aa_match = 0
            for j in range(min(pep_true_aa_len, pep_pred_aa_len)):
                if(pep_true_aa[j]==pep_pred_aa[j]):
                    aa_match+=1
            pep_match = 0
            if(pep_true==pep_pred):
                pep_match=1

            pep_eval_arr.append([pep_true_aa_len, pep_pred_aa_len, aa_match, pep_match])

        total_pep_num = len(pep_eval_arr)
        evaluations = [0 for _ in range(len(pep_eval_arr[0]))]
        for i in range(total_pep_num):
            for j in range(len(pep_eval_arr[i])):
                evaluations[j] += pep_eval_arr[i][j]

        aa_precision = evaluations[2]/(evaluations[1] + 1e-8)
        aa_recall = evaluations[2]/(evaluations[0] + 1e-8)
        pep_precision = evaluations[3]/(total_pep_num + 1e-8)

        #print(self.__class__.__name__+ ' ' + sys._getframe().f_code.co_name + ' ended '+ '+'*100)
        return([aa_precision, aa_recall, pep_precision])

    def my_finish_beams(
        self,
        tokens: torch.Tensor,
        precursors: torch.Tensor,
        step: int,
        tail_mass: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Track all beams that have been finished, either by predicting the stop
        token or because they were terminated due to exceeding the precursor
        m/z tolerance.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        step : int
            Index of the current decoding step.

        Returns
        -------
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams have been
            finished.
        beam_fits_precursor: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating if current beams are within precursor m/z
            tolerance.
        discarded_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams should be
            discarded (e.g. because they were predicted to end but violate the
            minimum peptide length).
        """

        # Check for tokens with a negative mass (i.e. neutral loss).
        aa_neg_mass = [None]
        for aa, mass in self.peptide_mass_calculator.masses.items():
            if mass < 0:
                aa_neg_mass.append(aa)

        # Find N-terminal residues.
        n_term = torch.Tensor(
            [
                self.decoder._aa2idx[aa]
                for aa in self.peptide_mass_calculator.masses
                if aa.startswith(("+", "-"))
            ]
        ).to(self.decoder.device)

        beam_fits_precursor = torch.zeros(
            tokens.shape[0], dtype=torch.bool
        ).to(self.encoder.device)

        # Beams with a stop token predicted in the current step can be finished.
        finished_beams = torch.zeros(tokens.shape[0], dtype=torch.bool).to(
            self.encoder.device
        )
        ends_stop_token = tokens[:, step] == self.stop_token
        finished_beams[ends_stop_token] = True

        # Beams with a dummy token predicted in the current step can be
        # discarded.
        discarded_beams = torch.zeros(tokens.shape[0], dtype=torch.bool).to(
            self.encoder.device
        )
        discarded_beams[tokens[:, step] == 0] = True

        # Discard beams with invalid modification combinations (i.e. N-terminal
        # modifications occur multiple times or in internal positions).
        if step > 1:  # Only relevant for longer predictions.
            dim0 = torch.arange(tokens.shape[0])
            final_pos = torch.full((ends_stop_token.shape[0],), step)
            final_pos[ends_stop_token] = step - 1

            # Multiple N-terminal modifications.
            multiple_mods = torch.isin(
                tokens[dim0, final_pos], n_term
            ) & torch.isin(tokens[dim0, final_pos - 1], n_term)

            # N-terminal modifications occur at an internal position.
            # Broadcasting trick to create a two-dimensional mask.
            mask = (final_pos - 1)[:, None] >= torch.arange(tokens.shape[1])

            internal_mods = torch.isin(
                torch.where(mask.to(self.encoder.device), tokens, 0), n_term
            ).any(dim=1)

            discarded_beams[multiple_mods | internal_mods] = True

        # Check which beams should be terminated or discarded based on the
        # predicted peptide.
        for i in range(len(finished_beams)):
            # Skip already discarded beams.
            if discarded_beams[i]:
                continue
            pred_tokens = tokens[i][: step + 1]
            #self._utils.parse_var(pred_tokens)
            peptide_len = len(pred_tokens)
            #self._utils.parse_var(peptide_len)
            peptide = self.decoder.detokenize(pred_tokens)
            #self._utils.parse_var(peptide)

            # Omit stop token.
            if self.decoder.reverse and peptide[0] == "$":
                peptide = peptide[1:]
                peptide_len -= 1
            elif not self.decoder.reverse and peptide[-1] == "$":
                peptide = peptide[:-1]
                peptide_len -= 1
            # Discard beams that were predicted to end but don't fit the minimum
            # peptide length.
            if finished_beams[i] and peptide_len < self.min_peptide_len:
                discarded_beams[i] = True
                continue
            # Terminate the beam if it has not been finished by the model but
            # the peptide mass exceeds the precursor m/z to an extent that it
            # cannot be corrected anymore by a subsequently predicted AA with
            # negative mass.
            precursor_mass = precursors[i, 0]
            precursor_charge = precursors[i, 1]
            precursor_mz = precursors[i, 2]
            matches_precursor_mz = exceeds_precursor_mz = False
            for aa in [None] if finished_beams[i] else aa_neg_mass:
                if aa is None:
                    calc_peptide = peptide
                else:
                    calc_peptide = peptide.copy()
                    calc_peptide.append(aa)
                try:
                    calc_mass = self.calculate_peptide_mass(
                        seq=calc_peptide, charge=precursor_charge
                    )

                    delta_mass_ppm = [
                        self.calculate_mass_error(
                            calc_mass,
                            tail_mass,
                            precursor_charge,
                            isotope,
                        )
                        for isotope in range(
                            self.isotope_error_range[0],
                            self.isotope_error_range[1] + 1,
                        )
                    ]

                    # Terminate the beam if the calculated m/z for the predicted
                    # peptide (without potential additional AAs with negative
                    # mass) is within the precursor m/z tolerance.
                    matches_precursor_mz = aa is None and any(
                        abs(d) < self.precursor_mass_tol
                        for d in delta_mass_ppm
                    )
                    #self._utils.parse_var(matches_precursor_mz)

                    # Terminate the beam if the calculated m/z exceeds the
                    # precursor m/z + tolerance and hasn't been corrected by a
                    # subsequently predicted AA with negative mass.
                    if matches_precursor_mz:
                        exceeds_precursor_mz = False
                    else:
                        exceeds_precursor_mz = all(
                            d > self.precursor_mass_tol for d in delta_mass_ppm
                        )
                        exceeds_precursor_mz = (
                            finished_beams[i] or aa is not None
                        ) and exceeds_precursor_mz
                    if matches_precursor_mz or exceeds_precursor_mz:
                        break
                except KeyError:
                    matches_precursor_mz = exceeds_precursor_mz = False
            # Finish beams that fit or exceed the precursor m/z.
            # Don't finish beams that don't include a stop token if they don't
            # exceed the precursor m/z tolerance yet.
            if(exceeds_precursor_mz):
                finished_beams[i] = True
            beam_fits_precursor[i] = matches_precursor_mz

        return finished_beams, beam_fits_precursor, discarded_beams

    def calculate_mass_error(self, calc_mass: float, obs_mass: float, charge=None, isotope: int = 0) -> float:
        return (calc_mass - (obs_mass - isotope * self._proton)) / obs_mass * 10**6

    def calculate_aa_pep_score(self, aa_scores: np.ndarray, fits_precursor_mz: bool) -> Tuple[np.ndarray, float]:
        #self._utils.parse_var(aa_scores, 'C')
        #self._utils.parse_var(np.isnan(aa_scores), 'np.isnan(aa_scores)')
        peptide_score = np.mean(aa_scores, where= ~np.isnan(aa_scores))
        #self._utils.parse_var(peptide_score, 'A')
        aa_scores = (aa_scores + peptide_score) / 2
        #self._utils.parse_var(aa_scores)

        if not fits_precursor_mz:
            peptide_score -= 1

        #self._utils.parse_var(peptide_score, 'B')

        return(aa_scores, peptide_score)

    def calculate_peptide_mass(self, seq, charge=None):
        calc_mass = sum([self._mass_dict[aa] for aa in seq])
        return calc_mass

    def my_cache_finished_beams(
        self,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        step: int,
        beams_to_cache: torch.Tensor,
        beam_fits_precursor: torch.Tensor,
        pred_cache: Dict[int, List[Tuple[float, float, np.ndarray, torch.Tensor]]],
    ):
        """
        Cache terminated beams.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        step : int
            Index of the current decoding step.
        beams_to_cache : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are ready for
            caching.
        beam_fits_precursor: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the beams are within the
            precursor m/z tolerance.
        pred_cache : Dict[int, List[Tuple[float, float, np.ndarray, torch.Tensor]]]
            Priority queue with finished beams for each spectrum, ordered by
            peptide score. For each finished beam, a tuple with the (negated)
            peptide score, amino acid-level scores, and the predicted tokens is
            stored.
        """
        beams_to_cache |= beam_fits_precursor
        for i in range(len(beams_to_cache)):
            if not beams_to_cache[i]:
                continue
            # Find the starting index of the spectrum.
            spec_idx = i // self.n_beams
            # FIXME: The next 3 lines are very similar as what's done in
            #  _finish_beams. Avoid code duplication?
            pred_tokens = tokens[i][: step + 1]
            # Omit the stop token from the peptide sequence (if predicted).
            has_stop_token = pred_tokens[-1] == self.stop_token
            pred_peptide = pred_tokens[:-1] if has_stop_token else pred_tokens
            # Don't cache this peptide if it was already predicted previously.
            if any(
                torch.equal(pred_cached[-1], pred_peptide)
                for pred_cached in pred_cache[spec_idx]
            ):
                # TODO: Add duplicate predictions with their highest score.
                continue
            smx = self.softmax(scores[i : i + 1, : step + 1, :])

            aa_scores = smx[0, range(len(pred_tokens)), pred_tokens].tolist()
            #self._utils.parse_var(aa_scores, 'A')

            # Add an explicit score 0 for the missing stop token in case this
            # was not predicted (i.e. early stopping).
            #if not has_stop_token:
            #    aa_scores.append(0)
            aa_scores = np.asarray(aa_scores)

            # Calculate the updated amino acid-level and the peptide scores.
            aa_scores, peptide_score = self.calculate_aa_pep_score(
                aa_scores, beam_fits_precursor[i]
            )

            # Omit the stop token from the amino acid-level scores.
            #aa_scores = aa_scores[:-1]

            # Add the prediction to the cache (minimum priority queue, maximum
            # the number of beams elements).

            if len(pred_cache[spec_idx]) < self.n_beams:
                heapadd = heapq.heappush
            else:
                heapadd = heapq.heappushpop
            heapadd(
                pred_cache[spec_idx],
                (
                    peptide_score,
                    np.random.random_sample(),
                    aa_scores,
                    torch.clone(pred_peptide),
                ),
            )
'''

###---------------------------------------------NEW-----------------------------------------------------------
    def _process_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        spectra, precursors, peptides = batch
        mzs = spectra[:,:,0]
        intensities = spectra[:,:,1]

        print(peptides)
        #tokens = self.tokenizer.tokenize(peptides)
        tokens = self.tokenizer.tokenize(peptides, add_stop=True)

        return(mzs, intensities, precursors, tokens)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Predict peptide sequences for a batch of MS/MS spectra.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, and
            ``precursor_charge``, each pointing to tensors with the
            corresponding data. The ``seq`` key is optional and
            contains the peptide sequences for training.

        Returns
        -------
        pred_peptides : List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide predictions.
            A peptide prediction consists of a tuple with the peptide
            score, the amino acid scores, and the predicted peptide
            sequence.
        """
        mzs, ints, precursors, tokens = self._process_batch(batch)

        print('\nmzs',end=':')
        print(mzs.shape)
        pp.pprint(mzs)

        print('\nints:')
        print(ints.shape)
        pp.pprint(ints)

        print('\nprecursors:')
        print(precursors.shape)
        pp.pprint(precursors)

        print('\ntokens:')
        print(tokens.shape)
        pp.pprint(tokens)

        #return self.beam_search_decode(mzs, ints, precursors)


    def _forward_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward learning step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, and
            ``precursor_charge``, each pointing to tensors with the
            corresponding data. The ``seq`` key is optional and
            contains the peptide sequences for training.

        Returns
        -------
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        tokens : torch.Tensor of shape (n_spectra, length)
            The predicted tokens for each spectrum.
        """
        mzs, ints, precursors, tokens = self._process_batch(batch)

        print('\nmzs',end=':')
        print(mzs.shape)
        pp.pprint(mzs)

        print('\nints:')
        print(ints.shape)
        pp.pprint(ints)

        print('\nprecursors:')
        print(precursors.shape)
        pp.pprint(precursors)

        print('\ntokens:')
        print(tokens.shape)
        pp.pprint(tokens)


        memories, mem_masks = self.encoder(mzs, ints)
        scores = self.decoder(
            tokens=tokens,
            memory=memories,
            memory_key_padding_mask=mem_masks,
            precursors=precursors,
        )
        return scores, tokens

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        *args,
        mode: str = "train",
    ) -> torch.Tensor:
        """
        A single training step.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            A batch from the SpectrumDataset, which contains keys:
            ``mz_array``, ``intensity_array``, ``precursor_mz``, and
            ``precursor_charge``, each pointing to tensors with the
            corresponding data. The ``seq`` key is optional and
            contains the peptide sequences for training.
        mode : str
            Logging key to describe the current stage.

        Returns
        -------
        torch.Tensor
            The loss of the training step.
        """
        pred, truth = self._forward_step(batch)
        pred = pred[:, :-1, :].reshape(-1, self.vocab_size)

        if mode == "train":
            loss = self.celoss(pred, truth.flatten())
        else:
            loss = self.val_celoss(pred, truth.flatten())
        self.log(
            f"{mode}_CELoss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=pred.shape[0],
        )
        return loss

    def on_train_start(self):
        """Log optimizer settings."""
        self.log("hp/optimizer_warmup_iters", self.warmup_iters)
        self.log(
            "hp/optimizer_cosine_schedule_period_iters",
            self.cosine_schedule_period_iters,
        )

    def on_train_epoch_end(self) -> None:
        """
        Log the training loss at the end of each epoch.
        """
        if "train_CELoss" in self.trainer.callback_metrics:
            train_loss = (
                self.trainer.callback_metrics["train_CELoss"].detach().item()
            )
        else:
            train_loss = np.nan
        metrics = {"step": self.trainer.global_step, "train": train_loss}
        self._history.append(metrics)
        self._log_history()