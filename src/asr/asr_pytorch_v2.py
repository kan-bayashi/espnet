#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import copy
import json
import logging
import math
import os
import pickle

# chainer related
import chainer
from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions

import torch

# spnet related
from asr_utils import adadelta_eps_decay
from asr_utils import CompareValueTrigger
from asr_utils import converter_kaldi
from asr_utils import delete_feat
from asr_utils import restore_snapshot
from e2e_asr_attctc_th import E2E
from e2e_asr_attctc_th import lecun_normal_init_parameters
from e2e_asr_attctc_th import Loss
from e2e_asr_attctc_th import set_forget_bias_to_one

# for kaldi io
import lazy_io

# numpy related
import matplotlib
matplotlib.use('Agg')


class PytorchSeqEvaluaterKaldi(extensions.Evaluator):
    '''Custom evaluater with Kaldi reader for pytorch'''

    def __init__(self, model, iterator, target, reader, device):
        super(PytorchSeqEvaluaterKaldi, self).__init__(
            iterator, target, device=device)
        self.reader = reader
        self.model = model

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

        summary = reporter_module.DictSummary()

        self.model.eval()
        with torch.no_grad():
            for batch in it:
                observation = {}
                with reporter_module.report_scope(observation):
                    # read scp files
                    # x: original json with loaded features
                    #    will be converted to chainer variable later
                    # batch only has one minibatch utterance, which is specified by batch[0]
                    x = converter_kaldi(batch[0], self.reader)
                    self.model(x)
                    delete_feat(x)
                summary.add(observation)
        self.model.train()

        return summary.compute_mean()


class PytorchSeqUpdaterKaldi(training.StandardUpdater):
    '''Custom updater with Kaldi reader for pytorch'''

    def __init__(self, model, grad_clip_threshold, train_iter, optimizer, reader, device):
        super(PytorchSeqUpdaterKaldi, self).__init__(
            train_iter, optimizer, device=None)
        self.model = model
        self.reader = reader
        self.grad_clip_threshold = grad_clip_threshold
        self.num_gpu = len(device)

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get the next batch ( a list of json files)
        batch = train_iter.__next__()

        # read scp files
        # x: original json with loaded features
        #    will be converted to chainer variable later
        # batch only has one minibatch utterance, which is specified by batch[0]
        if len(batch[0]) < self.num_gpu:
            logging.warning('batch size is less than number of gpus. Ignored')
            return
        x = converter_kaldi(batch[0], self.reader)

        # Compute the loss at this time step and accumulate it
        loss = 1. / self.num_gpu * self.model(x)
        optimizer.zero_grad()  # Clear the parameter gradients
        if self.num_gpu > 1:
            loss.backward(torch.ones(self.num_gpu))  # Backprop
        else:
            loss.backward()  # Backprop
        loss.detach()  # Truncate the graph
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_threshold)
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()
        delete_feat(x)


class DataParallel(torch.nn.DataParallel):
    def scatter(self, inputs, kwargs, device_ids, dim):
        r"""Scatter with support for kwargs dictionary"""
        if len(inputs) == 1:
            inputs = inputs[0]
        avg = int(math.ceil(len(inputs) * 1. / len(device_ids)))
        # inputs = scatter(inputs, device_ids, dim) if inputs else []
        inputs = [[inputs[i:i + avg]] for i in range(0, len(inputs), avg)]
        kwargs = torch.nn.scatter(kwargs, device_ids, dim) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        inputs = tuple(inputs)
        kwargs = tuple(kwargs)
        return inputs, kwargs

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids, self.dim)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)


def make_batchset(data, batch_size, max_length_in, max_length_out,
                  num_batches=0, batch_sort_key=None):
    minibatch = []
    # get feature dict
    feat_data = list(filter(lambda data: int(data[1]['idim']) != 320, data.items()))
    state_data = list(filter(lambda data: int(data[1]['idim']) == 320, data.items()))
    logging.info('# utts with feature vector: ' + str(len(feat_data)))
    logging.info('# utts with hidden state: ' + str(len(state_data)))
    for data in [feat_data, state_data]:
        if len(data) == 0:
            continue
        start = 0
        # sort it by output lengths (long to short)
        sorted_data = sorted(data, key=lambda d: int(d[1]['ilen']), reverse=True)
        # change batchsize depending on the input and output length
        while True:
            ilen = int(sorted_data[start][1]['ilen'])  # reverse
            olen = int(sorted_data[start][1]['olen'])  # reverse
            factor = max(int(ilen / max_length_in), int(olen / max_length_out))
            # if ilen = 1000 and max_length_in = 800
            # then b = batchsize / 2
            # and max(1, .) avoids batchsize = 0
            b = max(1, int(batch_size / (1 + factor)))
            end = min(len(sorted_data), start + b)
            minibatch.append(sorted_data[start:end])
            if end == len(sorted_data):
                break
            start = end

    # for debugging
    if num_batches > 0:
        minibatch = minibatch[:num_batches]
    logging.info('# minibatches: ' + str(len(minibatch)))

    return minibatch


def retrain(args):
    '''Run training'''
    # seed setting
    torch.manual_seed(args.seed)

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # by considering reproducability
    # revmoe type check
    if args.debugmode < 2:
        chainer.config.type_check = False
        logging.info('torch type check is disabled')
    # use determinisitic computation or not
    if args.debugmode < 1:
        torch.backends.cudnn.deterministic = False
        logging.info('torch cudnn deterministic is disabled')
    else:
        torch.backends.cudnn.deterministic = True

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # read training config
    with open(args.model_conf, "rb") as f:
        logging.info('reading a model config file from' + args.model_conf)
        idim, odim, train_args = pickle.load(f)

    # specify model architecture
    train_args.input_layer_idx = args.input_layer_idx
    e2e = E2E(idim, odim, train_args)
    model = Loss(e2e, train_args.mtlalpha)
    model.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

    if args.flatstart:
        # initialzie parameters of attention and decoder
        logging.info("training start from initialized value.")
        lecun_normal_init_parameters(model.predictor.dec)
        model.predictor.dec.embed.weight.data.normal_(0, 1)
        for l in range(len(model.predictor.dec.decoder)):
            set_forget_bias_to_one(model.predictor.dec.decoder[l].bias_ih)
    elif args.freeze_attention:
        logging.info("attention layer parameters are freezed.")
        for p in model.predictor.dec.att.parameters():
            p.requires_grad = False

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    # Set gpu
    reporter = model.reporter
    ngpu = args.ngpu
    if ngpu == 1:
        gpu_id = range(ngpu)
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()
    elif ngpu > 1:
        gpu_id = range(ngpu)
        logging.info('gpu id: ' + str(gpu_id))
        model = DataParallel(model, device_ids=gpu_id)
        model.cuda()
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu
    else:
        gpu_id = [-1]

    # Setup an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            params, rho=0.95, eps=args.eps)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr)
    elif args.opt == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # read json data
    with open(args.train_label, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_label, 'rb') as f:
        valid_json = json.load(f)['utts']

    # make minibatch list (variable length)
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches)
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches)
    # hack to make batchsze argument as 1
    # actual bathsize is included in a list
    train_iter = chainer.iterators.SerialIterator(train, 1)
    valid_iter = chainer.iterators.SerialIterator(
        valid, 1, repeat=False, shuffle=False)

    # prepare Kaldi reader
    train_reader = lazy_io.read_dict_scp(args.train_feat)
    valid_reader = lazy_io.read_dict_scp(args.valid_feat)

    # Set up a trainer
    updater = PytorchSeqUpdaterKaldi(
        model, args.grad_clip, train_iter, optimizer, train_reader, gpu_id)
    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    # Resume from a snapshot
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
        if ngpu > 1:
            model.module.load_state_dict(torch.load(args.outdir + '/model.acc.best'))
        else:
            model.load_state_dict(torch.load(args.outdir + '/model.acc.best'))
        model = trainer.updater.model

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(PytorchSeqEvaluaterKaldi(
        model, valid_iter, reporter, valid_reader, device=gpu_id))

    # Take a snapshot for each specified epoch
    trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                          'main/loss_ctc', 'validation/main/loss_ctc',
                                          'main/loss_att', 'validation/main/loss_att'],
                                         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/acc', 'validation/main/acc'],
                                         'epoch', file_name='acc.png'))

    # Save best models
    def torch_save(path, _):
        if ngpu > 1:
            torch.save(model.module.state_dict(), path)
            torch.save(model.module, path + ".pkl")
        else:
            torch.save(model.state_dict(), path)
            torch.save(model, path + ".pkl")

    trainer.extend(extensions.snapshot_object(model, 'model.loss.best', savefun=torch_save),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))
    trainer.extend(extensions.snapshot_object(model, 'model.acc.best', savefun=torch_save),
                   trigger=training.triggers.MaxValueTrigger('validation/main/acc'))

    # epsilon decay in the optimizer
    def torch_load(path, obj):
        if ngpu > 1:
            model.module.load_state_dict(torch.load(path))
        else:
            model.load_state_dict(torch.load(path))
        return obj
    if args.opt == 'adadelta':
        if args.criterion == 'acc':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
        elif args.criterion == 'loss':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(100, 'iteration')))
    report_keys = ['epoch', 'iteration', 'main/loss', 'main/loss_ctc', 'main/loss_att',
                   'validation/main/loss', 'validation/main/loss_ctc', 'validation/main/loss_att',
                   'main/acc', 'validation/main/acc', 'elapsed_time']
    if args.opt == 'adadelta':
        trainer.extend(extensions.observe_value(
            'eps', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["eps"]),
            trigger=(100, 'iteration'))
        report_keys.append('eps')
    trainer.extend(extensions.PrintReport(
        report_keys), trigger=(100, 'iteration'))

    trainer.extend(extensions.ProgressBar())

    # Run the training
    trainer.run()
