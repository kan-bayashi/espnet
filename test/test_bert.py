#!/usr/bin/env python

import numpy as np
import pytest
import torch

from argparse import Namespace

from espnet.nets.pytorch_backend.e2e_tts_bert import pad_list
from espnet.nets.pytorch_backend.e2e_tts_bert import Tacotron2
from espnet.nets.pytorch_backend.e2e_tts_bert import Tacotron2Loss


def make_model_args(**kwargs):
    defaults = dict(
        use_speaker_embedding=False,
        spk_embed_dim=None,
        aux_dim=32,
        embed_dim=512,
        elayers=1,
        eunits=512,
        econv_layers=3,
        econv_filts=5,
        econv_chans=512,
        dlayers=2,
        dunits=1024,
        prenet_layers=2,
        prenet_units=256,
        postnet_layers=5,
        postnet_filts=5,
        postnet_chans=512,
        output_activation=None,
        atype="forward_ta",
        adim=128,
        aconv_chans=32,
        aconv_filts=15,
        aux_adim=128,
        aux_aconv_chans=32,
        aux_aconv_filts=15,
        cumulate_att_w=True,
        use_batch_norm=True,
        use_concate=True,
        dropout=0.5,
        zoneout=0.1,
        reduction_factor=1,
        threshold=0.5,
        maxlenratio=5.0,
        minlenratio=0.0,
        use_cbhg=False,
        spc_dim=None,
        cbhg_conv_bank_layers=8,
        cbhg_conv_bank_chans=128,
        cbhg_conv_proj_filts=3,
        cbhg_conv_proj_chans=256,
        cbhg_highway_layers=4,
        cbhg_highway_units=128,
        cbhg_gru_units=256
    )
    defaults.update(kwargs)
    return defaults


def make_loss_args(**kwargs):
    defaults = dict(
        use_masking=True,
        bce_pos_weight=1.0,
        bce_lambda=1.0,
        use_guided_att=True,
        guided_att_lambda=1.0,
        guided_att_sigma=0.4
    )
    defaults.update(kwargs)
    return defaults


def make_inference_args(**kwargs):
    defaults = dict(
        threshold=0.5,
        maxlenratio=5.0,
        minlenratio=0.0
    )
    defaults.update(kwargs)
    return defaults


def prepare_inputs(bs, idim, odim, aux_dim, maxin_len, maxout_len,
                   spk_embed_dim=None, spc_dim=None):
    ilens = np.sort(np.random.randint(3, maxin_len, bs))[::-1].tolist()
    alens = [ilen // 3 + 3 for ilen in ilens]
    olens = np.sort(np.random.randint(3, maxout_len, bs))[::-1].tolist()

    ilens = torch.LongTensor(ilens)
    alens = torch.LongTensor(alens)
    olens = torch.LongTensor(olens)

    xs = [np.random.randint(0, idim, l) for l in ilens]
    auxs = [np.random.randn(l, aux_dim) for l in alens]
    ys = [np.random.randn(l, odim) for l in olens]

    xs = pad_list([torch.from_numpy(x).long() for x in xs], 0)
    auxs = pad_list([torch.from_numpy(aux).float() for aux in auxs], 0)
    ys = pad_list([torch.from_numpy(y).float() for y in ys], 0)

    labels = ys.new_zeros(ys.size(0), ys.size(1))
    for i, l in enumerate(olens):
        labels[i, l - 1:] = 1
    if spk_embed_dim is not None:
        spembs = torch.from_numpy(np.random.randn(bs, spk_embed_dim)).float()
    else:
        spembs = None
    if spc_dim is not None:
        spcs = [np.random.randn(l, spc_dim) for l in olens]
        spcs = pad_list([torch.from_numpy(spc).float() for spc in spcs], 0)
    else:
        spcs = None

    return xs, ilens, auxs, alens, ys, labels, olens, spembs, spcs


@pytest.mark.parametrize(
    "model_dict, loss_dict", [
        ({}, {}),
        ({"reduction_factor": 3}, {}),
        ({}, {"use_guided_att": True}),
    ])
def test_tacotron2_trainable_and_decodable(model_dict, loss_dict):
    # make args
    model_args = make_model_args(**model_dict)
    loss_args = make_loss_args(**loss_dict)
    inference_args = make_inference_args()

    # setup batch
    bs = 2
    maxin_len = 10
    maxout_len = 10
    idim = 5
    aux_dim = model_args['aux_dim']
    odim = 10
    if model_args['use_cbhg']:
        model_args['spc_dim'] = 129
    if model_args['use_speaker_embedding']:
        model_args['spk_embed_dim'] = 128
    batch = prepare_inputs(bs, idim, odim, aux_dim, maxin_len, maxout_len,
                           model_args['spk_embed_dim'], model_args['spc_dim'])
    xs, ilens, auxs, alens, ys, labels, olens, spembs, spcs = batch

    # define model
    model = Tacotron2(idim, odim, Namespace(**model_args))
    criterion = Tacotron2Loss(model, **loss_args)
    optimizer = torch.optim.Adam(model.parameters())

    # trainable
    loss = criterion(xs, ilens, auxs, alens, ys, labels, olens, spembs, spcs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # decodable
    model.eval()
    with torch.no_grad():
        spemb = None if model_args['spk_embed_dim'] is None else spembs[0]
        model.inference(xs[0][:ilens[0]], auxs[0][:alens[0]], Namespace(**inference_args), spemb)
        att_ws, aux_att_ws = model.calculate_all_attentions(xs, ilens, auxs, alens, ys, spembs)
