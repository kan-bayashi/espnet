#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import logging
import six

import chainer
import numpy as np
import torch
import torch.nn.functional as F

from numba import jit
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from espnet.nets.pytorch_backend.attentions import AttForward
from espnet.nets.pytorch_backend.attentions import AttForwardTA
from espnet.nets.pytorch_backend.attentions import AttLoc
from espnet.nets.pytorch_backend.nets_utils import to_device


def encoder_init(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain('relu'))


def decoder_init(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain('tanh'))


def make_non_pad_mask(lengths):
    """Function to make tensor mask containing indices of the non-padded part

    e.g.: lengths = [5, 3, 2]
          mask = [[1, 1, 1, 1 ,1],
                  [1, 1, 1, 0, 0],
                  [1, 1, 0, 0, 0]]

    :param list lengths: list of lengths (B)
    :return: mask tensor containing indices of non-padded part (B, Tmax)
    :rtype: torch.Tensor
    """
    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    maxlen = int(max(lengths))
    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    return seq_range_expand < seq_length_expand


class Reporter(chainer.Chain):
    def report(self, dicts):
        for d in dicts:
            chainer.reporter.report(d, self)


class ZoneOutCell(torch.nn.Module):
    """ZoneOut Cell

    This code is modified from https://github.com/eladhoffer/seq2seq.pytorch

    :param torch.nn.Module cell: pytorch recurrent cell
    :param float zoneout_prob: probability of zoneout
    """

    def __init__(self, cell, zoneout_prob=0.1):
        super(ZoneOutCell, self).__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout_prob = zoneout_prob
        if zoneout_prob > 1.0 or zoneout_prob < 0.0:
            raise ValueError("zoneout probability must be in the range from 0.0 to 1.0.")

    def forward(self, inputs, hidden):
        next_hidden = self.cell(inputs, hidden)
        next_hidden = self._zoneout(hidden, next_hidden, self.zoneout_prob)
        return next_hidden

    def _zoneout(self, h, next_h, prob):
        # apply recursively
        if isinstance(h, tuple):
            num_h = len(h)
            if not isinstance(prob, tuple):
                prob = tuple([prob] * num_h)
            return tuple([self._zoneout(h[i], next_h[i], prob[i]) for i in range(num_h)])

        if self.training:
            mask = h.new(*h.size()).bernoulli_(prob)
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h


@jit('f4[:, :, :](i8[:], i8[:], f4)', nopython=True)
def _make_guided_attention(ilens, olens, sigma):
    n_batch = ilens.shape[0]
    max_in = np.max(ilens)
    max_out = np.max(olens)
    gs = np.zeros((n_batch, max_out, max_in), dtype=np.float32)
    for b, (ilen, olen) in enumerate(zip(ilens, olens)):
        for i in range(ilen):
            for j in range(olen):
                gs[b, j, i] = 1 - np.exp(-(i / ilen - j / olen)**2 / (2 * (sigma ** 2)))

    return gs


@jit('u1[:, :, :](i8[:], i8[:])', nopython=True)
def _make_mask(ilens, olens):
    n_batch = ilens.shape[0]
    max_in = np.max(ilens)
    max_out = np.max(olens)
    mask = np.zeros((n_batch, max_out, max_in), dtype=np.uint8)
    for b, (ilen, olen) in enumerate(zip(ilens, olens)):
        mask[b, :olen, :ilen] = 1

    return mask


class GuidedAttentionLoss(torch.nn.Module):
    """Guided attention loss function

    :param float sigma: standard deviation to control how close attention to a diagonal
    """

    def __init__(self, sigma=0.4):
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = np.float32(sigma)

    def forward(self, att_ws, ilens, olens):
        ilens = np.array(list(map(int, ilens)), dtype=np.int64)
        olens = np.array(list(map(int, olens)), dtype=np.int64)
        gs = torch.from_numpy(_make_guided_attention(ilens, olens, self.sigma)).to(att_ws.device)
        masks = torch.from_numpy(_make_mask(ilens, olens)).to(att_ws.device)
        gs = gs.masked_select(masks)
        att_ws = att_ws.masked_select(masks)

        return torch.mean(gs * att_ws)


class Tacotron2Loss(torch.nn.Module):
    """Tacotron2 loss function

    :param torch.nn.Module model: tacotron2 model
    :param Namespace args: argments containing following attributes
        (bool) use_masking: whether to mask padded part in loss calculation
        (float) bce_pos_weight: weight of positive sample of stop token (only for use_masking=True)
        (float) bce_lambda: lambda value for binary cross entropy
        (bool) use_guided_att: whether to use guided attention
        (float) guided_att_lambda: lambda value for guided attention
        (float) guided_att_sigma: sigma value for guided attention
    """

    def __init__(self, model, args):
        super(Tacotron2Loss, self).__init__()
        self.model = model
        self.use_masking = args.use_masking
        self.bce_pos_weight = args.bce_pos_weight
        self.bce_lambda = args.bce_lambda
        self.use_guided_att = args.use_guided_att
        if self.use_guided_att:
            self.guided_att_lambda = args.guided_att_lambda
            self.guided_att_sigma = args.guided_att_sigma
            self.guided_att_loss = GuidedAttentionLoss(self.guided_att_sigma)
        if hasattr(model, 'module'):
            self.use_cbhg = model.module.use_cbhg
            self.reduction_factor = model.module.reduction_factor
        else:
            self.use_cbhg = model.use_cbhg
            self.reduction_factor = model.reduction_factor
        self.reporter = Reporter()

    def forward(self, xs, ilens, auxs, alens, ys, labels, olens, spembs=None, spcs=None):
        """Tacotron2 loss forward computation

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each input batch (B)
        :param torch.Tensor auxs: batch of the sequence of auxiliary features (B, Rmax, V)
        :param list alens: list of lengths of each auxiliary input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor labels: batch of the sequences of stop token labels (B, Lmax)
        :param list olens: batch of the lengths of each target (B)
        :param torch.Tensor spembs: batch of speaker embedding vector (B, spk_embed_dim)
        :param torch.Tensor spcs: batch of padded target features (B, Lmax, spc_dim)
        :return: loss value
        :rtype: torch.Tensor
        """
        # calcuate outputs
        if self.use_cbhg:
            cbhg_outs, after_outs, before_outs, logits, att_ws, aux_att_ws = self.model(
                xs, ilens, auxs, alens, ys, olens, spembs)
        else:
            after_outs, before_outs, logits, att_ws, aux_att_ws = self.model(
                xs, ilens, auxs, alens, ys, olens, spembs)

        # remove mod part
        if self.reduction_factor > 1:
            olens = [olen - olen % self.reduction_factor for olen in olens]
            ys = ys[:, :max(olens)]
            labels = labels[:, :max(olens)]
            spcs = spcs[:, :max(olens)] if spcs is not None else None

        # prepare weight of positive samples in cross entorpy
        if self.bce_pos_weight != 1.0:
            weights = ys.new(*labels.size()).fill_(1)
            weights.masked_fill_(labels.eq(1), self.bce_pos_weight)
        else:
            weights = None

        # perform masking for padded values
        if self.use_masking:
            mask = to_device(self, make_non_pad_mask(olens).unsqueeze(-1))
            ys = ys.masked_select(mask)
            after_outs = after_outs.masked_select(mask)
            before_outs = before_outs.masked_select(mask)
            labels = labels.masked_select(mask[:, :, 0])
            logits = logits.masked_select(mask[:, :, 0])
            weights = weights.masked_select(mask[:, :, 0]) if weights is not None else None
            if self.use_cbhg:
                spcs = spcs.masked_select(mask)
                cbhg_outs = cbhg_outs.masked_select(mask)

        # calculate main loss
        loss_reports = []
        l1_loss = F.l1_loss(after_outs, ys) + F.l1_loss(before_outs, ys)
        mse_loss = F.mse_loss(after_outs, ys) + F.mse_loss(before_outs, ys)
        bce_loss = self.bce_lambda * F.binary_cross_entropy_with_logits(logits, labels, weights)
        loss = l1_loss + mse_loss + bce_loss

        # add reports
        loss_reports += [
            {'l1_loss': l1_loss.item()},
            {'mse_loss': mse_loss.item()},
            {'bce_loss': bce_loss.item()}
        ]

        # calculate cbhg loss
        if self.use_cbhg:
            cbhg_l1_loss = F.l1_loss(cbhg_outs, spcs)
            cbhg_mse_loss = F.mse_loss(cbhg_outs, spcs)
            loss += cbhg_l1_loss + cbhg_mse_loss
            loss_reports += [
                {'cbhg_l1_loss': cbhg_l1_loss.item()},
                {'cbhg_mse_loss': cbhg_mse_loss.item()}
            ]

        # calculate guided attention loss
        if self.use_guided_att:
            olens = [olen // self.reduction_factor for olen in olens]
            att_loss = self.guided_att_lambda * self.guided_att_loss(att_ws, ilens, olens)
            aux_att_loss = self.guided_att_lambda * self.guided_att_loss(aux_att_ws, alens, olens)
            loss += att_loss + aux_att_loss
            loss_reports += [
                {'aux_att_loss': aux_att_loss.item()},
                {'att_loss': att_loss.item()},
            ]

        # reports
        loss_reports += [
            {'loss': loss.item()},
        ]
        self.reporter.report(loss_reports)

        return loss


class Tacotron2(torch.nn.Module):
    """Tacotron2 based Seq2Seq converts chars to features

    :param int idim: dimension of the inputs
    :param int odim: dimension of the outputs
    :param Namespace args: argments containing following attributes
        (int) spk_embed_dim: dimension of the speaker embedding
        (int) embed_dim: dimension of character embedding
        (int) elayers: the number of encoder blstm layers
        (int) eunits: the number of encoder blstm units
        (int) econv_layers: the number of encoder conv layers
        (int) econv_filts: the number of encoder conv filter size
        (int) econv_chans: the number of encoder conv filter channels
        (int) dlayers: the number of decoder lstm layers
        (int) dunits: the number of decoder lstm units
        (int) prenet_layers: the number of prenet layers
        (int) prenet_units: the number of prenet units
        (int) postnet_layers: the number of postnet layers
        (int) postnet_filts: the number of postnet filter size
        (int) postnet_chans: the number of postnet filter channels
        (str) output_activation: the name of activation function for outputs
        (int) adim: the number of dimension of mlp in attention
        (int) aconv_chans: the number of attention conv filter channels
        (int) aconv_filts: the number of attention conv filter size
        (bool) cumulate_att_w: whether to cumulate previous attention weight
        (bool) use_batch_norm: whether to use batch normalization
        (bool) use_concate: whether to concatenate encoder embedding with decoder lstm outputs
        (float) dropout: dropout rate
        (float) zoneout: zoneout rate
        (int) reduction_factor: reduction factor
        (bool) use_cbhg: whether to use CBHG module
        (int) cbhg_conv_bank_layers: the number of convoluional banks in CBHG
        (int) cbhg_conv_bank_chans: the number of channels of convolutional bank in CBHG
        (int) cbhg_proj_filts: the number of filter size of projection layeri in CBHG
        (int) cbhg_proj_chans: the number of channels of projection layer in CBHG
        (int) cbhg_highway_layers: the number of layers of highway network in CBHG
        (int) cbhg_highway_units: the number of units of highway network in CBHG
        (int) cbhg_gru_units: the number of units of GRU in CBHG
    """

    def __init__(self, idim, odim, args):
        super(Tacotron2, self).__init__()
        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.spk_embed_dim = args.spk_embed_dim
        self.embed_dim = args.embed_dim
        self.elayers = args.elayers
        self.eunits = args.eunits
        self.econv_layers = args.econv_layers
        self.econv_filts = args.econv_filts
        self.econv_chans = args.econv_chans
        self.dlayers = args.dlayers
        self.dunits = args.dunits
        self.prenet_layers = args.prenet_layers
        self.prenet_units = args.prenet_units
        self.postnet_layers = args.postnet_layers
        self.postnet_chans = args.postnet_chans
        self.postnet_filts = args.postnet_filts
        self.atype = args.atype
        self.adim = args.adim
        self.aconv_filts = args.aconv_filts
        self.aconv_chans = args.aconv_chans
        self.aux_dim = args.aux_dim
        self.aux_proj_dim = args.aux_proj_dim if hasattr(args, "aux_proj_dim") else None
        self.aux_adim = args.aux_adim
        self.aux_aconv_filts = args.aux_aconv_filts
        self.aux_aconv_chans = args.aux_aconv_chans
        self.cumulate_att_w = args.cumulate_att_w
        self.use_batch_norm = args.use_batch_norm
        self.use_concate = args.use_concate
        self.dropout = args.dropout
        self.zoneout = args.zoneout
        self.reduction_factor = args.reduction_factor
        self.use_cbhg = args.use_cbhg
        if self.use_cbhg:
            self.spc_dim = args.spc_dim
            self.cbhg_conv_bank_layers = args.cbhg_conv_bank_layers
            self.cbhg_conv_bank_chans = args.cbhg_conv_bank_chans
            self.cbhg_conv_proj_filts = args.cbhg_conv_proj_filts
            self.cbhg_conv_proj_chans = args.cbhg_conv_proj_chans
            self.cbhg_highway_layers = args.cbhg_highway_layers
            self.cbhg_highway_units = args.cbhg_highway_units
            self.cbhg_gru_units = args.cbhg_gru_units

        # define activation function for the final output
        if args.output_activation is None:
            self.output_activation_fn = None
        elif hasattr(F, args.output_activation):
            self.output_activation_fn = getattr(F, args.output_activation)
        else:
            raise ValueError('there is no such an activation function. (%s)' % args.output_activation)
        # define network modules
        self.enc = Encoder(idim=self.idim,
                           embed_dim=self.embed_dim,
                           elayers=self.elayers,
                           eunits=self.eunits,
                           econv_layers=self.econv_layers,
                           econv_chans=self.econv_chans,
                           econv_filts=self.econv_filts,
                           use_batch_norm=self.use_batch_norm,
                           dropout=self.dropout)
        if self.aux_proj_dim is not None:
            self.aux_proj = torch.nn.Linear(self.aux_dim, self.aux_proj_dim)
        dec_idim = self.eunits if self.spk_embed_dim is None else self.eunits + self.spk_embed_dim
        if self.atype == "location":
            att = AttLoc(dec_idim,
                         self.dunits,
                         self.adim,
                         self.aconv_chans,
                         self.aconv_filts)
            aux_att = AttLoc(self.aux_proj_dim if self.aux_proj_dim is not None else self.aux_dim,
                             self.dunits,
                             self.aux_adim,
                             self.aux_aconv_chans,
                             self.aux_aconv_filts)
        elif self.atype == "forward":
            att = AttForward(dec_idim,
                             self.dunits,
                             self.adim,
                             self.aconv_chans,
                             self.aconv_filts)
            aux_att = AttForward(self.aux_proj_dim if self.aux_proj_dim is not None else self.aux_dim,
                                 self.dunits,
                                 self.aux_adim,
                                 self.aux_aconv_chans,
                                 self.aux_aconv_filts)
            if self.cumulate_att_w:
                logging.warning("cumulation of attention weights is disabled in forward attention.")
                self.cumulate_att_w = False
        elif self.atype == "forward_ta":
            att = AttForwardTA(dec_idim,
                               self.dunits,
                               self.adim,
                               self.aconv_chans,
                               self.aconv_filts,
                               self.odim)
            aux_att = AttForwardTA(self.aux_proj_dim if self.aux_proj_dim is not None else self.aux_dim,
                                   self.dunits,
                                   self.aux_adim,
                                   self.aux_aconv_chans,
                                   self.aux_aconv_filts,
                                   self.odim)
            if self.cumulate_att_w:
                logging.warning("cumulation of attention weights is disabled in forward attention.")
                self.cumulate_att_w = False
        else:
            raise NotImplementedError("Support only location or forward")
        dec_idim = dec_idim + self.aux_proj_dim if self.aux_proj_dim is not None else dec_idim + self.aux_dim
        self.dec = Decoder(idim=dec_idim,
                           odim=self.odim,
                           att=att,
                           aux_att=aux_att,
                           dlayers=self.dlayers,
                           dunits=self.dunits,
                           prenet_layers=self.prenet_layers,
                           prenet_units=self.prenet_units,
                           postnet_layers=self.postnet_layers,
                           postnet_chans=self.postnet_chans,
                           postnet_filts=self.postnet_filts,
                           output_activation_fn=self.output_activation_fn,
                           cumulate_att_w=self.cumulate_att_w,
                           use_batch_norm=self.use_batch_norm,
                           use_concate=self.use_concate,
                           dropout=self.dropout,
                           zoneout=self.zoneout,
                           reduction_factor=self.reduction_factor)
        if self.use_cbhg:
            self.cbhg = CBHG(idim=self.odim,
                             odim=self.spc_dim,
                             conv_bank_layers=self.cbhg_conv_bank_layers,
                             conv_bank_chans=self.cbhg_conv_bank_chans,
                             conv_proj_filts=self.cbhg_conv_proj_filts,
                             conv_proj_chans=self.cbhg_conv_proj_chans,
                             highway_layers=self.cbhg_highway_layers,
                             highway_units=self.cbhg_highway_units,
                             gru_units=self.cbhg_gru_units)

        # initialize
        self.enc.apply(encoder_init)
        self.dec.apply(decoder_init)

    def forward(self, xs, ilens, auxs, alens, ys, olens=None, spembs=None):
        """Tacotron2 forward computation

        :param torch.Tensor xs1: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each input batch (B)
        :param torch.Tensor auxs: batch of the sequence of auxiliary features (B, Rmax, V)
        :param list alens: list of lengths of each auxiliary input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor olens:
        :param torch.Tensor spembs: batch of speaker embedding vector (B, spk_embed_dim)
        :return: outputs with postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: outputs without postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: stop logits (B, Lmax)
        :rtype: torch.Tensor
        :return: attention weights (B, Lmax, Tmax)
        :rtype: torch.Tensor
        :return: aux attention weights (B, Rmax, Tmax)
        :rtype: torch.Tensor
        """
        # check ilens type (should be list of int)
        if isinstance(ilens, torch.Tensor) or isinstance(ilens, np.ndarray):
            ilens = list(map(int, ilens))
        if isinstance(alens, torch.Tensor) or isinstance(alens, np.ndarray):
            alens = list(map(int, alens))

        hs, hlens = self.enc(xs, ilens)
        if self.spk_embed_dim is not None:
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = torch.cat([hs, spembs], dim=-1)

        if self.aux_proj_dim is not None:
            auxs = self.aux_proj(auxs)

        # check the use of data_parallel
        if xs.shape[1] != max(ilens):
            n_pad = xs.shape[1] - max(ilens)
        else:
            n_pad = 0
        if auxs.shape[1] != max(alens):
            aux_n_pad = auxs.shape[1] - max(alens)
            auxs = auxs[:, :max(alens)]
        else:
            aux_n_pad = 0

        after_outs, before_outs, logits, att_ws, aux_att_ws = self.dec(hs, hlens, auxs, alens, ys)

        # pad to concate outputs with data_parallel
        if n_pad != 0:
            att_ws = F.pad(att_ws, (0, n_pad), "constant", 0)
        if aux_n_pad != 0:
            aux_att_ws = F.pad(aux_att_ws, (0, aux_n_pad), "constant", 0)

        if self.use_cbhg:
            if self.reduction_factor > 1:
                olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            cbhg_outs, _ = self.cbhg(after_outs, olens)
            return cbhg_outs, after_outs, before_outs, logits, att_ws, aux_att_ws
        else:
            return after_outs, before_outs, logits, att_ws, aux_att_ws

    def inference(self, x, aux, inference_args, spemb=None):
        """Generates the sequence of features given the sequences of characters

        :param torch.Tensor x: the sequence of characters (T)
        :param Namespace inference_args: argments containing following attributes
            (float) threshold: threshold in inference
            (float) minlenratio: minimum length ratio in inference
            (float) maxlenratio: maximum length ratio in inference
        :param torch.Tensor spemb: speaker embedding vector (spk_embed_dim)
        :return: the sequence of features (L, odim)
        :rtype: torch.Tensor
        :return: the sequence of stop probabilities (L)
        :rtype: torch.Tensor
        :return: the sequence of attention weight (L, T)
        :rtype: torch.Tensor
        """
        # get options
        threshold = inference_args.threshold
        minlenratio = inference_args.minlenratio
        maxlenratio = inference_args.maxlenratio

        # inference
        h = self.enc.inference(x)
        if self.aux_proj_dim is not None:
            aux = self.aux_proj(aux)
        if self.spk_embed_dim is not None:
            spemb = F.normalize(spemb, dim=0).unsqueeze(0).expand(h.size(0), -1)
            h = torch.cat([h, spemb], dim=-1)
        outs, probs, att_ws, aux_att_ws = self.dec.inference(h, aux, threshold, minlenratio, maxlenratio)

        if self.use_cbhg:
            cbhg_outs = self.cbhg.inference(outs)
            return cbhg_outs, probs, att_ws, aux_att_ws
        else:
            return outs, probs, att_ws, aux_att_ws

    def calculate_all_attentions(self, xs, ilens, auxs, alens, ys, spembs=None):
        """Tacotron2 forward computation

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param torch.Tensor ilens: list of lengths of each input batch (B)
        :param torch.Tensor auxs: batch of the sequence of auxiliary features (B, Rmax, V)
        :param list alens: list of lengths of each auxiliary input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor spembs: batch of speaker embedding vector (B, spk_embed_dim)
        :return: attention weights (B, Lmax, Tmax)
        :rtype: numpy array
        """
        # check ilens type (should be list of int)
        if isinstance(ilens, torch.Tensor) or isinstance(ilens, np.ndarray):
            ilens = list(map(int, ilens))

        self.eval()
        with torch.no_grad():
            hs, hlens = self.enc(xs, ilens)
            if self.spk_embed_dim is not None:
                spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
                hs = torch.cat([hs, spembs], dim=-1)
            att_ws, aux_att_ws = self.dec.calculate_all_attentions(hs, hlens, auxs, alens, ys)
        self.train()

        return att_ws.cpu().numpy(), aux_att_ws.cpu().numpy()


class Encoder(torch.nn.Module):
    """Character embedding encoder

    This is the encoder which converts the sequence of characters into
    the sequence of hidden states. The network structure is based on
    that of tacotron2 in the field of speech synthesis.

    :param int idim: dimension of the inputs
    :param int embed_dim: dimension of character embedding
    :param int elayers: the number of encoder blstm layers
    :param int eunits: the number of encoder blstm units
    :param int econv_layers: the number of encoder conv layers
    :param int econv_filts: the number of encoder conv filter size
    :param int econv_chans: the number of encoder conv filter channels
    :param bool use_batch_norm: whether to use batch normalization
    :param float dropout: dropout rate
    """

    def __init__(self, idim,
                 embed_dim=512,
                 elayers=1,
                 eunits=512,
                 econv_layers=3,
                 econv_chans=512,
                 econv_filts=5,
                 use_batch_norm=True,
                 use_residual=False,
                 dropout=0.5):
        super(Encoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.embed_dim = embed_dim
        self.elayers = elayers
        self.eunits = eunits
        self.econv_layers = econv_layers
        self.econv_chans = econv_chans if econv_layers != 0 else -1
        self.econv_filts = econv_filts if econv_layers != 0 else -1
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.dropout = dropout
        # define network layer modules
        self.embed = torch.nn.Embedding(self.idim, self.embed_dim)
        if self.econv_layers > 0:
            self.convs = torch.nn.ModuleList()
            for layer in six.moves.range(self.econv_layers):
                ichans = self.embed_dim if layer == 0 else self.econv_chans
                if self.use_batch_norm:
                    self.convs += [torch.nn.Sequential(
                        torch.nn.Conv1d(ichans, self.econv_chans, self.econv_filts, stride=1,
                                        padding=(self.econv_filts - 1) // 2, bias=False),
                        torch.nn.BatchNorm1d(self.econv_chans),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(self.dropout))]
                else:
                    self.convs += [torch.nn.Sequential(
                        torch.nn.Conv1d(ichans, self.econv_chans, self.econv_filts, stride=1,
                                        padding=(self.econv_filts - 1) // 2, bias=False),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(self.dropout))]
        else:
            self.convs = None
        iunits = econv_chans if self.econv_layers != 0 else self.embed_dim
        self.blstm = torch.nn.LSTM(
            iunits, self.eunits // 2, self.elayers,
            batch_first=True,
            bidirectional=True)

    def forward(self, xs, ilens):
        """Character encoding forward computation

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param list ilens: list of lengths of each batch (B)
        :return: batch of sequences of padded encoder states (B, Tmax, eunits)
        :rtype: torch.Tensor
        :return: batch of lengths of each encoder states (B)
        :rtype: list
        """
        xs = self.embed(xs).transpose(1, 2)
        for l in six.moves.range(self.econv_layers):
            if self.use_residual:
                xs += self.convs[l](xs)
            else:
                xs = self.convs[l](xs)
        xs = pack_padded_sequence(xs.transpose(1, 2), ilens, batch_first=True)
        self.blstm.flatten_parameters()
        xs, _ = self.blstm(xs)  # (B, Tmax, C)
        xs, hlens = pad_packed_sequence(xs, batch_first=True)

        return xs, hlens

    def inference(self, x):
        """Character encoder inference

        :param torch.Tensor x: the sequence of character ids (T)
        :return: the sequence encoder states (T, eunits)
        :rtype: torch.Tensor
        """
        assert len(x.size()) == 1
        xs = x.unsqueeze(0)
        ilens = [x.size(0)]

        return self.forward(xs, ilens)[0][0]


class Decoder(torch.nn.Module):
    """Decoder to predict the sequence of features

    This the decoder which generate the sequence of features from
    the sequence of the hidden states. The network structure is
    based on that of the tacotron2 in the field of speech synthesis.

    :param int idim: dimension of the inputs
    :param int odim: dimension of the outputs
    :param instance att: instance of attention class
    :param int dlayers: the number of decoder lstm layers
    :param int dunits: the number of decoder lstm units
    :param int prenet_layers: the number of prenet layers
    :param int prenet_units: the number of prenet units
    :param int postnet_layers: the number of postnet layers
    :param int postnet_filts: the number of postnet filter size
    :param int postnet_chans: the number of postnet filter channels
    :param function output_activation_fn: activation function for outputs
    :param bool cumulate_att_w: whether to cumulate previous attention weight
    :param bool use_batch_norm: whether to use batch normalization
    :param bool use_concate: whether to concatenate encoder embedding with decoder lstm outputs
    :param float dropout: dropout rate
    :param float zoneout: zoneout rate
    :param int reduction_factor: reduction factor
    :param float threshold: threshold in inference
    :param float minlenratio: minimum length ratio in inference
    :param float maxlenratio: maximum length ratio in inference
    """

    def __init__(self, idim, odim, att, aux_att,
                 dlayers=2,
                 dunits=1024,
                 prenet_layers=2,
                 prenet_units=256,
                 postnet_layers=5,
                 postnet_chans=512,
                 postnet_filts=5,
                 output_activation_fn=None,
                 cumulate_att_w=True,
                 use_batch_norm=True,
                 use_concate=True,
                 dropout=0.5,
                 zoneout=0.1,
                 threshold=0.5,
                 reduction_factor=1,
                 maxlenratio=5.0,
                 minlenratio=0.0):
        super(Decoder, self).__init__()
        # store the hyperparameters
        self.idim = idim
        self.odim = odim
        self.att = att
        self.aux_att = aux_att
        self.dlayers = dlayers
        self.dunits = dunits
        self.prenet_layers = prenet_layers
        self.prenet_units = prenet_units if prenet_layers != 0 else self.odim
        self.postnet_layers = postnet_layers
        self.postnet_chans = postnet_chans if postnet_layers != 0 else -1
        self.postnet_filts = postnet_filts if postnet_layers != 0 else -1
        self.output_activation_fn = output_activation_fn
        self.cumulate_att_w = cumulate_att_w
        self.use_batch_norm = use_batch_norm
        self.use_concate = use_concate
        self.dropout = dropout
        self.zoneout = zoneout
        self.reduction_factor = reduction_factor
        self.threshold = threshold
        self.maxlenratio = maxlenratio
        self.minlenratio = minlenratio
        # check attention type
        if isinstance(self.att, AttForwardTA):
            self.use_att_extra_inputs = True
        else:
            self.use_att_extra_inputs = False
        # define lstm network
        self.lstm = torch.nn.ModuleList()
        for layer in six.moves.range(self.dlayers):
            iunits = self.idim + self.prenet_units if layer == 0 else self.dunits
            lstm = torch.nn.LSTMCell(iunits, self.dunits)
            if zoneout > 0.0:
                lstm = ZoneOutCell(lstm, self.zoneout)
            self.lstm += [lstm]
        # define prenet
        if self.prenet_layers > 0:
            self.prenet = torch.nn.ModuleList()
            for layer in six.moves.range(self.prenet_layers):
                ichans = self.odim if layer == 0 else self.prenet_units
                self.prenet += [torch.nn.Sequential(
                    torch.nn.Linear(ichans, self.prenet_units, bias=False),
                    torch.nn.ReLU())]
        else:
            self.prenet = None
        # define postnet
        if self.postnet_layers > 0:
            self.postnet = torch.nn.ModuleList()
            for layer in six.moves.range(self.postnet_layers - 1):
                ichans = self.odim if layer == 0 else self.postnet_chans
                ochans = self.odim if layer == self.postnet_layers - 1 else self.postnet_chans
                if use_batch_norm:
                    self.postnet += [torch.nn.Sequential(
                        torch.nn.Conv1d(ichans, ochans, self.postnet_filts, stride=1,
                                        padding=(self.postnet_filts - 1) // 2, bias=False),
                        torch.nn.BatchNorm1d(ochans),
                        torch.nn.Tanh(),
                        torch.nn.Dropout(self.dropout))]
                else:
                    self.postnet += [torch.nn.Sequential(
                        torch.nn.Conv1d(ichans, ochans, self.postnet_filts, stride=1,
                                        padding=(self.postnet_filts - 1) // 2, bias=False),
                        torch.nn.Tanh(),
                        torch.nn.Dropout(self.dropout))]
            ichans = self.postnet_chans if self.postnet_layers != 1 else self.odim
            if use_batch_norm:
                self.postnet += [torch.nn.Sequential(
                    torch.nn.Conv1d(ichans, odim, self.postnet_filts, stride=1,
                                    padding=(self.postnet_filts - 1) // 2, bias=False),
                    torch.nn.BatchNorm1d(odim),
                    torch.nn.Dropout(self.dropout))]
            else:
                self.postnet += [torch.nn.Sequential(
                    torch.nn.Conv1d(ichans, odim, self.postnet_filts, stride=1,
                                    padding=(self.postnet_filts - 1) // 2, bias=False),
                    torch.nn.Dropout(self.dropout))]
        else:
            self.postnet = None
        # define projection layers
        iunits = self.idim + self.dunits if self.use_concate else self.dunits
        self.feat_out = torch.nn.Linear(iunits, self.odim * self.reduction_factor, bias=False)
        self.prob_out = torch.nn.Linear(iunits, self.reduction_factor)

    def zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.dunits)
        return init_hs

    def forward(self, hs, hlens, auxs, alens, ys):
        """Decoder forward computation

        :param torch.Tensor hs: batch of the sequences of padded hidden states (B, Tmax, idim)
        :param list hlens: list of lengths of each input batch (B)
        :param torch.Tensor auxs: batch of the sequence of auxiliary features (B, Rmax, V)
        :param list alens: list of lengths of each auxiliary input batch (B)
        :param torch.Tensor ys: batch of the sequences of padded target features (B, Lmax, odim)
        :return: outputs with postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: outputs without postnets (B, Lmax, odim)
        :rtype: torch.Tensor
        :return: stop logits (B, Lmax)
        :rtype: torch.Tensor
        :return: attention weights (B, Lmax, Tmax)
        :rtype: torch.Tensor
        :return: aux attention weights (B, Rmax, Tmax)
        :rtype: torch.Tensor
        """
        # thin out frames (B, Lmax, odim) ->  (B, Lmax/r, odim)
        if self.reduction_factor > 1:
            ys = ys[:, self.reduction_factor - 1::self.reduction_factor]

        # length list should be list of int
        hlens = list(map(int, hlens))
        alens = list(map(int, alens))

        # initialize hidden states of decoder
        c_list = [self.zero_state(hs)]
        z_list = [self.zero_state(hs)]
        for _ in six.moves.range(1, self.dlayers):
            c_list += [self.zero_state(hs)]
            z_list += [self.zero_state(hs)]
        prev_out = hs.new_zeros(hs.size(0), self.odim)

        # initialize attention
        prev_att_w = None
        prev_aux_att_w = None
        self.att.reset()
        self.aux_att.reset()

        # loop for an output sequence
        outs, logits, att_ws, aux_att_ws = [], [], [], []
        for y in ys.transpose(0, 1):
            if self.use_att_extra_inputs:
                att_c, att_w = self.att(
                    hs, hlens, z_list[0], prev_att_w, prev_out)
                aux_att_c, aux_att_w = self.aux_att(
                    auxs, alens, z_list[0], prev_aux_att_w, prev_out)
            else:
                att_c, att_w = self.att(
                    hs, hlens, z_list[0], prev_att_w)
                aux_att_c, aux_att_w = self.aux_att(
                    auxs, alens, z_list[0], prev_aux_att_w)
            prenet_out = self._prenet_forward(prev_out)
            xs = torch.cat([att_c, aux_att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.lstm[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            zcs = torch.cat([z_list[-1], att_c, aux_att_c], dim=1) if self.use_concate else z_list[-1]
            outs += [self.feat_out(zcs).view(hs.size(0), self.odim, -1)]
            logits += [self.prob_out(zcs)]
            att_ws += [att_w]
            aux_att_ws += [aux_att_w]
            prev_out = y  # teacher forcing
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
                prev_aux_att_w = prev_aux_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w
                prev_aux_att_w = aux_att_w

        logits = torch.cat(logits, dim=1)  # (B, Lmax)
        before_outs = torch.cat(outs, dim=2)  # (B, odim, Lmax)
        att_ws = torch.stack(att_ws, dim=1)
        aux_att_ws = torch.stack(aux_att_ws, dim=1)

        if self.reduction_factor > 1:
            before_outs = before_outs.view(before_outs.size(0), self.odim, -1)  # (B, odim, Lmax)

        after_outs = before_outs + self._postnet_forward(before_outs)  # (B, odim, Lmax)
        before_outs = before_outs.transpose(2, 1)  # (B, Lmax, odim)
        after_outs = after_outs.transpose(2, 1)  # (B, Lmax, odim)
        logits = logits

        # apply activation function for scaling
        if self.output_activation_fn is not None:
            before_outs = self.output_activation_fn(before_outs)
            after_outs = self.output_activation_fn(after_outs)

        return after_outs, before_outs, logits, att_ws, aux_att_ws

    def inference(self, h, aux, threshold=0.5, minlenratio=0.0, maxlenratio=10.0):
        """Generate the sequence of features given the encoder hidden states

        :param torch.Tensor h: the sequence of encoder states (T, C)
        :param torch.Tensor aux: the sequence of bert features (R, V)
        :param float threshold: threshold in inference
        :param float minlenratio: minimum length ratio in inference
        :param float maxlenratio: maximum length ratio in inference
        :return: the sequence of features (L, D)
        :rtype: torch.Tensor
        :return: the sequence of stop probabilities (L)
        :rtype: torch.Tensor
        :return: the sequence of attention weight (L, T)
        :rtype: torch.Tensor
        """
        # setup
        assert len(h.size()) == 2
        assert len(aux.size()) == 2
        hs = h.unsqueeze(0)
        ilens = [h.size(0)]
        auxs = aux.unsqueeze(0)
        alens = [aux.size(0)]
        maxlen = int(h.size(0) * maxlenratio)
        minlen = int(h.size(0) * minlenratio)

        # initialize hidden states of decoder
        c_list = [self.zero_state(hs)]
        z_list = [self.zero_state(hs)]
        for _ in six.moves.range(1, self.dlayers):
            c_list += [self.zero_state(hs)]
            z_list += [self.zero_state(hs)]
        prev_out = hs.new_zeros(1, self.odim)

        # initialize attention
        prev_att_w = None
        prev_aux_att_w = None
        self.att.reset()
        self.aux_att.reset()

        # loop for an output sequence
        idx = 0
        outs, att_ws, aux_att_ws, probs = [], [], [], []
        while True:
            # updated index
            idx += self.reduction_factor

            # decoder calculation
            if self.use_att_extra_inputs:
                att_c, att_w = self.att(
                    hs, ilens, z_list[0], prev_att_w, prev_out)
                aux_att_c, aux_att_w = self.aux_att(
                    auxs, alens, z_list[0], prev_aux_att_w, prev_out)
            else:
                att_c, att_w = self.att(
                    hs, ilens, z_list[0], prev_att_w)
                aux_att_c, aux_att_w = self.aux_att(
                    auxs, alens, z_list[0], prev_aux_att_w)
            att_ws += [att_w]
            aux_att_ws += [aux_att_w]
            prenet_out = self._prenet_forward(prev_out)
            xs = torch.cat([att_c, aux_att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.lstm[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            zcs = torch.cat([z_list[-1], att_c, aux_att_c], dim=1) if self.use_concate else z_list[-1]
            outs += [self.feat_out(zcs).view(1, self.odim, -1)]  # [(1, odim, r), ...]
            probs += [torch.sigmoid(self.prob_out(zcs))[0]]  # [(r), ...]
            if self.output_activation_fn is not None:
                prev_out = self.output_activation_fn(outs[-1][:, :, -1])  # (1, odim)
            else:
                prev_out = outs[-1][:, :, -1]  # (1, odim)
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
                prev_aux_att_w = prev_aux_att_w + aux_att_w  # Note: error when use +=
            else:
                prev_att_w = att_w
                prev_aux_att_w = aux_att_w

            # check whether to finish generation
            if int(sum(probs[-1] >= threshold)) > 0 or idx >= maxlen:
                # check mininum length
                if idx < minlen:
                    continue
                outs = torch.cat(outs, dim=2)  # (1, odim, L)
                outs = outs + self._postnet_forward(outs)  # (1, odim, L)
                outs = outs.transpose(2, 1).squeeze(0)  # (L, odim)
                probs = torch.cat(probs, dim=0)
                att_ws = torch.cat(att_ws, dim=0)
                aux_att_ws = torch.cat(aux_att_ws, dim=0)
                break

        if self.output_activation_fn is not None:
            outs = self.output_activation_fn(outs)

        return outs, probs, att_ws, aux_att_ws

    def calculate_all_attentions(self, hs, hlens, auxs, alens, ys):
        """Decoder attention calculation

        :param torch.Tensor hs: batch of the sequences of padded hidden states (B, Tmax, idim)
        :param list hlens: list of lengths of each input batch (B)
        :param torch.Tensor auxs: batch of the sequence of auxiliary features (B, Rmax, V)
        :param list alens: list of lengths of each auxiliary input batch (B)
        :param torch.Tensor ys: batch of the sequences of padded target features (B, Lmax, odim)
        :return: attention weights (B, Lmax, Tmax)
        :rtype: numpy array
        """
        # thin out frames (B, Lmax, odim) ->  (B, Lmax/r, odim)
        if self.reduction_factor > 1:
            ys = ys[:, self.reduction_factor - 1::self.reduction_factor]

        # length list should be list of int
        hlens = list(map(int, hlens))
        alens = list(map(int, alens))

        # initialize hidden states of decoder
        c_list = [self.zero_state(hs)]
        z_list = [self.zero_state(hs)]
        for _ in six.moves.range(1, self.dlayers):
            c_list += [self.zero_state(hs)]
            z_list += [self.zero_state(hs)]
        prev_out = hs.new_zeros(hs.size(0), self.odim)

        # initialize attention
        prev_att_w = None
        prev_aux_att_w = None
        self.att.reset()
        self.aux_att.reset()

        # loop for an output sequence
        att_ws = []
        aux_att_ws = []
        for y in ys.transpose(0, 1):
            if self.use_att_extra_inputs:
                att_c, att_w = self.att(
                    hs, hlens, z_list[0], prev_att_w, prev_out)
                aux_att_c, aux_att_w = self.aux_att(
                    auxs, alens, z_list[0], prev_aux_att_w, prev_out)
            else:
                att_c, att_w = self.att(
                    hs, hlens, z_list[0], prev_att_w)
                aux_att_c, aux_att_w = self.aux_att(
                    auxs, alens, z_list[0], prev_aux_att_w)
            att_ws += [att_w]
            aux_att_ws += [aux_att_w]
            prenet_out = self._prenet_forward(prev_out)
            xs = torch.cat([att_c, aux_att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.lstm[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            prev_out = y  # teacher forcing
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
                prev_aux_att_w = prev_aux_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w
                prev_aux_att_w = aux_att_w

        att_ws = torch.stack(att_ws, dim=1)  # (B, Lmax, Tmax)
        aux_att_ws = torch.stack(aux_att_ws, dim=1)  # (B, Lmax, Tmax)

        return att_ws, aux_att_ws

    def _prenet_forward(self, x):
        if self.prenet is not None:
            for l in six.moves.range(self.prenet_layers):
                x = F.dropout(self.prenet[l](x), self.dropout)
        return x

    def _postnet_forward(self, xs):
        if self.postnet is not None:
            for l in six.moves.range(self.postnet_layers):
                xs = self.postnet[l](xs)
        return xs


class CBHG(torch.nn.Module):
    """CBHG module to convert log mel-fbank to linear spectrogram

    :param int idim: dimension of the inputs
    :param int odim: dimension of the outputs
    :param int conv_bank_layers: the number of convolution bank layers
    :param int conv_bank_chans: the number of channels in convolution bank
    :param int conv_proj_filts: kernel size of convolutional projection layer
    :param int conv_proj_chans: the number of channels in convolutional projection layer
    :param int highway_layers: the number of highway network layers
    :param int highway_units: the number of highway network units
    :param int gru_units: the number of GRU units (for both directions)
    """

    def __init__(self,
                 idim,
                 odim,
                 conv_bank_layers=8,
                 conv_bank_chans=128,
                 conv_proj_filts=3,
                 conv_proj_chans=256,
                 highway_layers=4,
                 highway_units=128,
                 gru_units=256):
        super(CBHG, self).__init__()
        self.idim = idim
        self.odim = odim
        self.conv_bank_layers = conv_bank_layers
        self.conv_bank_chans = conv_bank_chans
        self.conv_proj_filts = conv_proj_filts
        self.conv_proj_chans = conv_proj_chans
        self.highway_layers = highway_layers
        self.highway_units = highway_units
        self.gru_units = gru_units

        # define 1d convolution bank
        self.conv_bank = torch.nn.ModuleList()
        for k in range(1, self.conv_bank_layers + 1):
            if k % 2 != 0:
                padding = (k - 1) // 2
            else:
                padding = ((k - 1) // 2, (k - 1) // 2 + 1)
            self.conv_bank += [torch.nn.Sequential(
                torch.nn.ConstantPad1d(padding, 0.0),
                torch.nn.Conv1d(idim, self.conv_bank_chans, k, stride=1,
                                padding=0, bias=True),
                torch.nn.BatchNorm1d(self.conv_bank_chans),
                torch.nn.ReLU())]

        # define max pooling (need padding for one-side to keep same length)
        self.max_pool = torch.nn.Sequential(
            torch.nn.ConstantPad1d((0, 1), 0.0),
            torch.nn.MaxPool1d(2, stride=1))

        # define 1d convolution projection
        self.projections = torch.nn.Sequential(
            torch.nn.Conv1d(self.conv_bank_chans * self.conv_bank_layers, self.conv_proj_chans,
                            self.conv_proj_filts, stride=1,
                            padding=(self.conv_proj_filts - 1) // 2, bias=True),
            torch.nn.BatchNorm1d(self.conv_proj_chans),
            torch.nn.ReLU(),
            torch.nn.Conv1d(self.conv_proj_chans, self.idim,
                            self.conv_proj_filts, stride=1,
                            padding=(self.conv_proj_filts - 1) // 2, bias=True),
            torch.nn.BatchNorm1d(self.idim),
        )

        # define highway network
        self.highways = torch.nn.ModuleList()
        self.highways += [torch.nn.Linear(idim, self.highway_units)]
        for _ in range(self.highway_layers):
            self.highways += [HighwayNet(self.highway_units)]

        # define bidirectional GRU
        self.gru = torch.nn.GRU(self.highway_units, gru_units // 2, num_layers=1,
                                batch_first=True, bidirectional=True)

        # define final projection
        self.output = torch.nn.Linear(gru_units, odim, bias=True)

    def forward(self, xs, ilens):
        """CBHG module forward

        :param torch.Tensor xs: batch of the sequences of inputs (B, Tmax, idim)
        :param torch.Tensor ilens: list of lengths of each input batch (B)
        :return: batch of sequences of padded outputs (B, Tmax, eunits)
        :rtype: torch.Tensor
        :return: batch of lengths of each encoder states (B)
        :rtype: list
        """
        xs = xs.transpose(1, 2)  # (B, idim, Tmax)
        convs = []
        for k in range(self.conv_bank_layers):
            convs += [self.conv_bank[k](xs)]
        convs = torch.cat(convs, dim=1)  # (B, #CH * #BANK, Tmax)
        convs = self.max_pool(convs)
        convs = self.projections(convs).transpose(1, 2)  # (B, Tmax, idim)
        xs = xs.transpose(1, 2) + convs
        # + 1 for dimension adjustment layer
        for l in range(self.highway_layers + 1):
            xs = self.highways[l](xs)

        # sort by length
        xs, ilens, sort_idx = self._sort_by_length(xs, ilens)

        # total_length needs for DataParallel
        # (see https://github.com/pytorch/pytorch/pull/6327)
        total_length = xs.size(1)
        xs = pack_padded_sequence(xs, ilens, batch_first=True)
        self.gru.flatten_parameters()
        xs, _ = self.gru(xs)
        xs, ilens = pad_packed_sequence(xs, batch_first=True, total_length=total_length)

        # revert sorting by length
        xs, ilens = self._revert_sort_by_length(xs, ilens, sort_idx)

        xs = self.output(xs)  # (B, Tmax, odim)

        return xs, ilens

    def inference(self, x):
        """CBHG module inference

        :param torch.Tensor x: input (T, idim)
        :return: the sequence encoder states (T, odim)
        :rtype: torch.Tensor
        """
        assert len(x.size()) == 2
        xs = x.unsqueeze(0)
        ilens = x.new([x.size(0)]).long()

        return self.forward(xs, ilens)[0][0]

    def _sort_by_length(self, xs, ilens):
        sort_ilens, sort_idx = ilens.sort(0, descending=True)
        return xs[sort_idx], ilens[sort_idx], sort_idx

    def _revert_sort_by_length(self, xs, ilens, sort_idx):
        _, revert_idx = sort_idx.sort(0)
        return xs[revert_idx], ilens[revert_idx]


class HighwayNet(torch.nn.Module):
    """Highway Network

    :param int idim: dimension of the inputs
    """

    def __init__(self, idim):
        super(HighwayNet, self).__init__()
        self.idim = idim
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(idim, idim),
            torch.nn.ReLU())
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(idim, idim),
            torch.nn.Sigmoid())

    def forward(self, x):
        """Highway Network forward

        :param torch.Tensor x: batch of inputs (B, *, idim)
        :return: batch of outputs (B, *, idim)
        :rtype: torch.Tensor
        """
        proj = self.projection(x)
        gate = self.gate(x)
        return proj * gate + x * (1.0 - gate)
