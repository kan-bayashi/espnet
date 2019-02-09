#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import json
import logging
import math
import os

import chainer
import kaldiio
import numpy as np
import torch

from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_save
from espnet.asr.asr_utils import torch_snapshot
from espnet.nets.pytorch_backend.e2e_asr import pad_list
from espnet.nets.pytorch_backend.e2e_tts_bert import Tacotron2
from espnet.nets.pytorch_backend.e2e_tts_bert import Tacotron2Loss
from espnet.transform.transformation import using_transform_config
from espnet.tts.tts_utils import make_batchset
from espnet.utils.io_utils import LoadInputsAndTargets

from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.training.train_utils import check_early_stop

import matplotlib

from espnet.utils.training.tensorboard_logger import TensorboardLogger
from tensorboardX import SummaryWriter

matplotlib.use('Agg')

REPORT_INTERVAL = 100


class PlotAttentionReport(training.extension.Extension):
    """Plot attention reporter

    :param function att_vis_fn: function of attention visualization
    :param list data: list json utt key items
    :param str outdir: directory to save figures
    :param CustomConverter converter: function to convert data
    :param int | torch.device device: device
    :param bool reverse: If True, input and output length are reversed
    """

    def __init__(self, att_vis_fn, data, outdir, converter, device, reverse=False):
        self.att_vis_fn = att_vis_fn
        self.data = copy.deepcopy(data)
        self.outdir = outdir
        self.converter = converter
        self.device = device
        self.reverse = reverse
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def __call__(self, trainer):
        att_ws, aux_att_ws = self.get_attention_weights()
        for idx, att_w in enumerate(att_ws):
            filename = "%s/%s.ep.{.updater.epoch}.png" % (
                self.outdir, self.data[idx][0])
            att_w = self.get_attention_weight(idx, att_w)
            self._plot_and_save_attention(att_w, filename.format(trainer))
        for idx, aux_att_w in enumerate(aux_att_ws):
            filename = "%s/aux_%s.ep.{.updater.epoch}.png" % (
                self.outdir, self.data[idx][0])
            aux_att_w = self.get_attention_weight(idx, aux_att_w)
            self._plot_and_save_attention(aux_att_w, filename.format(trainer))

    def log_attentions(self, logger, step):
        att_ws, aux_att_ws = self.get_attention_weights()
        for idx, att_w in enumerate(att_ws):
            att_w = self.get_attention_weight(idx, att_w)
            plot = self.draw_attention_plot(att_w)
            logger.add_figure("%s" % (self.data[idx][0]), plot.gcf(), step)
            plot.clf()
        for idx, aux_att_w in enumerate(aux_att_ws):
            aux_att_w = self.get_attention_weight(idx, aux_att_w)
            plot = self.draw_attention_plot(aux_att_w)
            logger.add_figure("aux_%s" % (self.data[idx][0]), plot.gcf(), step)
            plot.clf()

    def get_attention_weights(self):
        batch = self.converter([self.converter.transform(self.data)], self.device)
        att_ws, aux_att_ws = self.att_vis_fn(*batch)
        return att_ws, aux_att_ws

    def get_attention_weight(self, idx, att_w):
        if self.reverse:
            dec_len = int(self.data[idx][1]['input'][0]['shape'][0])
            enc_len = int(self.data[idx][1]['output'][0]['shape'][0])
        else:
            dec_len = int(self.data[idx][1]['output'][0]['shape'][0])
            enc_len = int(self.data[idx][1]['input'][0]['shape'][0])
        if len(att_w.shape) == 3:
            att_w = att_w[:, :dec_len, :enc_len]
        else:
            att_w = att_w[:dec_len, :enc_len]
        return att_w

    def draw_attention_plot(self, att_w):
        import matplotlib.pyplot as plt
        if len(att_w.shape) == 3:
            for h, aw in enumerate(att_w, 1):
                plt.subplot(1, len(att_w), h)
                plt.imshow(aw, aspect="auto")
                plt.xlabel("Encoder Index")
                plt.ylabel("Decoder Index")
        else:
            plt.imshow(att_w, aspect="auto")
            plt.xlabel("Encoder Index")
            plt.ylabel("Decoder Index")
        plt.tight_layout()
        return plt

    def _plot_and_save_attention(self, att_w, filename):
        plt = self.draw_attention_plot(att_w)
        plt.savefig(filename)
        plt.close()


class CustomEvaluator(extensions.Evaluator):
    """Custom Evaluator for Tacotron2 training

    :param torch.nn.Model model : The model to evaluate
    :param chainer.dataset.Iterator iterator : The validation iterator
    :param target :
    :param CustomConverter converter : The batch converter
    :param torch.device device : The device to use
    """

    def __init__(self, model, iterator, target, converter, device):
        super(CustomEvaluator, self).__init__(iterator, target)
        self.model = model
        self.converter = converter
        self.device = device

    # The core part of the update routine can be customized by overriding.
    def evaluate(self):
        iterator = self._iterators['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = chainer.reporter.DictSummary()

        self.model.eval()
        with torch.no_grad():
            for batch in it:
                observation = {}
                with chainer.reporter.report_scope(observation):
                    # convert to torch tensor
                    x = self.converter(batch, self.device)
                    self.model(*x)
                summary.add(observation)
        self.model.train()

        return summary.compute_mean()


class CustomUpdater(training.StandardUpdater):
    """Custom updater for Tacotron2 training

    :param torch.nn.Module model: The model to update
    :param float grad_clip : The gradient clipping value to use
    :param chainer.dataset.Iterator train_iter : The training iterator
    :param optimizer :
    :param CustomConverter converter : The batch converter
    :param torch.device device : The device to use
    """

    def __init__(self, model, grad_clip, train_iter, optimizer, converter, device):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.grad_clip = grad_clip
        self.converter = converter
        self.device = device
        self.clip_grad_norm = torch.nn.utils.clip_grad_norm_

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch (a list of json files)
        batch = train_iter.next()
        x = self.converter(batch, self.device)

        # compute loss and gradient
        loss = self.model(*x)
        optimizer.zero_grad()
        loss.backward()

        # compute the gradient norm to check if it is normal or not
        grad_norm = self.clip_grad_norm(self.model.parameters(), self.grad_clip)
        logging.debug('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()


class CustomConverter(object):
    """Custom converter for Tacotron2 training

    :param bool return_targets:
    :param bool use_speaker_embedding:
    :param bool use_second_target:
    """

    def __init__(self,
                 return_targets=True,
                 use_speaker_embedding=False,
                 use_second_target=False,
                 preprocess_conf=None):
        self.return_targets = return_targets
        self.use_speaker_embedding = use_speaker_embedding
        self.use_second_target = use_second_target
        self.load_inputs_and_targets = LoadInputsAndTargets(
            mode='tts',
            use_speaker_embedding=use_speaker_embedding,
            use_second_target=use_second_target,
            preprocess_conf=preprocess_conf)

    def transform(self, item):
        # load batch
        xs, ys, spembs, spcs = self.load_inputs_and_targets(item)
        return xs, ys, spembs, spcs

    def __call__(self, batch, device):
        # batch should be located in list
        assert len(batch) == 1
        inputs_and_targets = batch[0]

        # parse inputs and targets
        xs, ys, spembs, auxs = inputs_and_targets

        # get list of lengths (must be tensor for DataParallel)
        ilens = torch.from_numpy(np.array([x.shape[0] for x in xs])).long().to(device)
        olens = torch.from_numpy(np.array([y.shape[0] for y in ys])).long().to(device)

        # perform padding and conversion to tensor
        xs = pad_list([torch.from_numpy(x).long() for x in xs], 0).to(device)
        ys = pad_list([torch.from_numpy(y).float() for y in ys], 0).to(device)

        # make labels for stop prediction
        labels = ys.new_zeros(ys.size(0), ys.size(1))
        for i, l in enumerate(olens):
            labels[i, l - 1:] = 1.0

        # load second target
        if auxs is not None:
            alens = torch.from_numpy(np.array([aux.shape[0] for aux in auxs])).long().to(device)
            auxs = pad_list([torch.from_numpy(aux).float() for aux in auxs], 0).to(device)

        # load speaker embedding
        if spembs is not None:
            spembs = torch.from_numpy(np.array(spembs)).float().to(device)

        if self.return_targets:
            return xs, ilens, auxs, alens, ys, labels, olens, spembs
        else:
            return xs, ilens, auxs, alens, ys, spembs


def train(args):
    """Train with the given args

    :param Namespace args: The program arguments
    """
    set_deterministic_pytorch(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())

    # reverse input and output dimension
    idim = int(valid_json[utts[0]]['output'][0]['shape'][1])
    odim = int(valid_json[utts[0]]['input'][0]['shape'][1])
    args.aux_dim = int(valid_json[utts[0]]['input'][1]['shape'][1])
    if args.use_cbhg:
        args.spc_dim = int(valid_json[utts[0]]['input'][1]['shape'][1])
    if args.use_speaker_embedding:
        args.spk_embed_dim = int(valid_json[utts[0]]['input'][1]['shape'][0])
    else:
        args.spk_embed_dim = None
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))
    logging.info('#aux dims: ' + str(args.aux_dim))

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to' + model_conf)
        f.write(json.dumps((idim, odim, vars(args)), indent=4, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # specify model architecture
    tacotron2 = Tacotron2(idim, odim, args)
    logging.info(tacotron2)

    # check the use of multi-gpu
    if args.ngpu > 1:
        tacotron2 = torch.nn.DataParallel(tacotron2, device_ids=list(range(args.ngpu)))
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    tacotron2 = tacotron2.to(device)

    # define loss
    model = Tacotron2Loss(tacotron2, args)
    reporter = model.reporter

    # Setup an optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), args.lr, eps=args.eps,
        weight_decay=args.weight_decay)

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, 'target', reporter)
    setattr(optimizer, 'serialize', lambda s: reporter.serialize(s))

    # Setup a converter
    converter = CustomConverter(return_targets=True,
                                use_speaker_embedding=args.use_speaker_embedding,
                                use_second_target=True,
                                preprocess_conf=args.preprocess_conf)

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    # make minibatch list (variable length)
    train_batchset = make_batchset(train_json, args.batch_size,
                                   args.maxlen_in, args.maxlen_out,
                                   args.minibatches, args.batch_sort_key,
                                   min_batch_size=args.ngpu if args.ngpu > 1 else 1)
    valid_batchset = make_batchset(valid_json, args.batch_size,
                                   args.maxlen_in, args.maxlen_out,
                                   args.minibatches, args.batch_sort_key,
                                   min_batch_size=args.ngpu if args.ngpu > 1 else 1)
    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    if args.n_iter_processes > 0:
        train_iter = chainer.iterators.MultiprocessIterator(
            TransformDataset(train_batchset, converter.transform),
            batch_size=1, n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
        valid_iter = chainer.iterators.MultiprocessIterator(
            TransformDataset(valid_batchset, converter.transform),
            batch_size=1, repeat=False, shuffle=False,
            n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
    else:
        train_iter = chainer.iterators.SerialIterator(
            TransformDataset(train_batchset, converter.transform),
            batch_size=1)
        valid_iter = chainer.iterators.SerialIterator(
            TransformDataset(valid_batchset, converter.transform),
            batch_size=1, repeat=False, shuffle=False)

    # Set up a trainer
    updater = CustomUpdater(model, args.grad_clip, train_iter, optimizer, converter, device)
    trainer = training.Trainer(updater, (args.epochs, 'epoch'), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(CustomEvaluator(model, valid_iter, reporter, converter, device))

    # Save snapshot for each epoch
    trainer.extend(torch_snapshot(), trigger=(1, 'epoch'))

    # Save best models
    trainer.extend(extensions.snapshot_object(tacotron2, 'model.loss.best', savefun=torch_save),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))

    # Save attention figure for each epoch
    if args.num_save_attention > 0:
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
        if hasattr(tacotron2, "module"):
            att_vis_fn = tacotron2.module.calculate_all_attentions
        else:
            att_vis_fn = tacotron2.calculate_all_attentions
        att_reporter = PlotAttentionReport(
            att_vis_fn, data, args.outdir + '/att_ws',
            converter=CustomConverter(return_targets=False,
                                      use_speaker_embedding=args.use_speaker_embedding,
                                      use_second_target=True,
                                      preprocess_conf=args.preprocess_conf),
            device=device, reverse=True)
        trainer.extend(att_reporter, trigger=(1, 'epoch'))
    else:
        att_reporter = None

    # Make a plot for training and validation values
    plot_keys = ['main/loss', 'validation/main/loss',
                 'main/l1_loss', 'validation/main/l1_loss',
                 'main/mse_loss', 'validation/main/mse_loss',
                 'main/bce_loss', 'validation/main/bce_loss']
    trainer.extend(extensions.PlotReport(['main/l1_loss', 'validation/main/l1_loss'],
                                         'epoch', file_name='l1_loss.png'))
    trainer.extend(extensions.PlotReport(['main/mse_loss', 'validation/main/mse_loss'],
                                         'epoch', file_name='mse_loss.png'))
    trainer.extend(extensions.PlotReport(['main/bce_loss', 'validation/main/bce_loss'],
                                         'epoch', file_name='bce_loss.png'))
    if args.use_guided_att:
        plot_keys += ['main/att_loss', 'validation/main/att_loss',
                      'main/aux_att_loss', 'validation/main/aux_att_loss']
        trainer.extend(extensions.PlotReport(['main/att_loss', 'validation/main/att_loss',
                                              'main/aux_att_loss', 'validation/main/aux_att_loss'],
                                             'epoch', file_name='att_loss.png'))
    if args.use_cbhg:
        plot_keys += ['main/cbhg_l1_loss', 'validation/main/cbhg_l1_loss',
                      'main/cbhg_mse_loss', 'validation/main/cbhg_mse_loss']
        trainer.extend(extensions.PlotReport(['main/cbhg_l1_loss', 'validation/main/cbhg_l1_loss'],
                                             'epoch', file_name='cbhg_l1_loss.png'))
        trainer.extend(extensions.PlotReport(['main/cbhg_mse_loss', 'validation/main/cbhg_mse_loss'],
                                             'epoch', file_name='cbhg_mse_loss.png'))
    trainer.extend(extensions.PlotReport(plot_keys, 'epoch', file_name='loss.png'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(REPORT_INTERVAL, 'iteration')))
    report_keys = plot_keys[:]
    report_keys[0:0] = ['epoch', 'iteration', 'elapsed_time']
    trainer.extend(extensions.PrintReport(report_keys), trigger=(REPORT_INTERVAL, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=REPORT_INTERVAL))

    if args.patience > 0:
        trainer.stop_trigger = chainer.training.triggers.EarlyStoppingTrigger(monitor=args.early_stop_criterion,
                                                                              patients=args.patience,
                                                                              max_trigger=(args.epochs, 'epoch'))
    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        writer = SummaryWriter(log_dir=args.tensorboard_dir)
        trainer.extend(TensorboardLogger(writer, att_reporter))

    # Run the training
    trainer.run()
    check_early_stop(trainer, args.epochs)


def decode(args):
    """Decode with the given args

    :param Namespace args: The program arguments
    """
    set_deterministic_pytorch(args)
    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # show arguments
    for key in sorted(vars(args).keys()):
        logging.info('args: ' + key + ': ' + str(vars(args)[key]))

    # define model
    tacotron2 = Tacotron2(idim, odim, train_args)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    torch_load(args.model, tacotron2)
    tacotron2.eval()

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    tacotron2 = tacotron2.to(device)

    # read json data
    with open(args.json, 'rb') as f:
        js = json.load(f)['utts']

    # check directory
    outdir = os.path.dirname(args.out)
    if len(outdir) != 0 and not os.path.exists(outdir):
        os.makedirs(outdir)

    load_inputs_and_targets = LoadInputsAndTargets(
        mode='tts', load_input=True, sort_in_input_length=False,
        use_speaker_embedding=train_args.use_speaker_embedding,
        use_second_target=True,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None else args.preprocess_conf)

    with torch.no_grad(), kaldiio.WriteHelper('ark,scp:{o}.ark,{o}.scp'.format(o=args.out)) as f:
        for idx, utt_id in enumerate(js.keys()):
            batch = [(utt_id, js[utt_id])]
            with using_transform_config({'train': False}):
                data = load_inputs_and_targets(batch)
            if train_args.use_speaker_embedding:
                spemb = data[2][0]
                spemb = torch.FloatTensor(spemb).to(device)
            else:
                spemb = None
            x = data[0][0]
            x = torch.LongTensor(x).to(device)
            aux = data[3][0]
            aux = torch.FloatTensor(aux).to(device)

            # decode and write
            outs, _, _, _ = tacotron2.inference(x, aux, args, spemb)
            if outs.size(0) == x.size(0) * args.maxlenratio:
                logging.warning("output length reaches maximum length (%s)." % utt_id)
            logging.info('(%d/%d) %s (size:%d->%d)' % (
                idx + 1, len(js.keys()), utt_id, x.size(0), outs.size(0)))
            f[utt_id] = outs.cpu().numpy()
