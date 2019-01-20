"""
This is the loadable seq2seq trainer library that is
in charge of training details, loss compute, and statistics.
See train.py for a use case of this library.

Note!!! To make this a general library, we implement *only*
mechanism things here(i.e. what to do), and leave the strategy
things to users(i.e. how to do it). Also see train.py(one of the
users of this library) for the strategy things we do.
"""
import os
import sys
import ipdb
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt
import onmt.io
import onmt.modules


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time

    def output(self, epoch, batch, n_batches, start):
        """Write out statistics to stdout.

        Args:
           epoch (int): current epoch
           batch (int): current batch
           n_batch (int): total batches
           start (int): start time of epoch.
        """
        t = self.elapsed_time()
        print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
               "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
              (epoch, batch,  n_batches,
               self.accuracy(),
               self.ppl(),
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log(self, prefix, experiment, lr):
        t = self.elapsed_time()
        experiment.add_scalar_value(prefix + "_ppl", self.ppl())
        experiment.add_scalar_value(prefix + "_accuracy", self.accuracy())
        experiment.add_scalar_value(prefix + "_tgtper",  self.n_words / t)
        experiment.add_scalar_value(prefix + "_lr", lr)


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.Model.NMTModel`): translation model to train
            train_iter: training data iterator
            valid_iter: validate data iterator
            train_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.Loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.Optim.Optim`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
    """
    def __init__(self,
                 model,
                 train_iter,
                 valid_iter,
                 tgt_vocab,
                 optim,
                 trunc_size=0,
                 shard_size=32,
                 data_type='text',
                 opt=None):
        # Basic attributes.
        self.model = model
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.opt = opt

        self.tgt_vocab_len = len(tgt_vocab)
        self.padding_idx = tgt_vocab.stoi[onmt.io.PAD_WORD]

        self.weight = torch.ones(self.tgt_vocab_len).cuda()
        self.weight[self.padding_idx] = 0

        # Set model in training mode.
        self.model.train()

    def train(self, epoch, report_func=None):
        """ Train next epoch.
        Args:
            epoch(int): the epoch number
            report_func(fn): function for logging

        Returns:
            stats (:obj:`onmt.Statistics`): epoch loss statistics
        """
        total_stats = Statistics()
        report_stats = Statistics()

        for i, batch in enumerate(self.train_iter):
            batch_size = batch.tgt.size(1)
            target_size = batch.tgt.size(0)
            # Truncated BPTT
            trunc_size = self.trunc_size if self.trunc_size else target_size

            dec_state = None
            src = onmt.io.make_features(batch, 'src', self.data_type)

            _, src_lengths = batch.src
            report_stats.n_src_words += src_lengths.sum()

            tgt = onmt.io.make_features(batch, 'tgt')

            if self.opt.encoder_model == 'Rev' and self.opt.decoder_model == 'Rev':
                if self.opt.use_reverse:
                    tgt_lengths = torch.tensor([tgt.size(0)], device=tgt.device)

                    self.model.zero_grad()
                    loss, num_words, num_correct, enc_output_dict, dec_output_dict, mem_dict = self.model.forward_and_backward(src,
                                                                                                                               tgt,
                                                                                                                               src_lengths,
                                                                                                                               tgt_lengths,
                                                                                                                               self.weight,
                                                                                                                               self.padding_idx)

                    loss_data = loss.data.clone()
                    batch_stats = onmt.Statistics(loss.item(), float(num_words), float(num_correct))

                    self.optim.step()

                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)
                else:
                    self.model.zero_grad()
                    tgt_lengths = torch.tensor([tgt.size(0)], device=tgt.device)
                    loss, num_words, num_correct, attn, enc_output_dict, dec_output_dict, mem_dict = self.model(src,
                                                                                                                tgt,
                                                                                                                src_lengths,
                                                                                                                tgt_lengths,
                                                                                                                self.weight,
                                                                                                                self.padding_idx)
                    loss_data = loss.data.clone()
                    # batch_stats = self._stats(loss_data, outputs.data, batch.tgt[1:batch.tgt.size(0)].view(-1).data)
                    # batch_stats = self._stats(loss_data, F.log_softmax(outputs.data), batch.tgt[1:batch.tgt.size(0)].view(-1).data)
                    batch_stats = onmt.Statistics(loss.item(), float(num_words), float(num_correct))

                    loss.div(batch_size).backward()
                    self.optim.step()

                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)
            else:
                self.model.zero_grad()
                model_output_tuple = self.model(src, tgt, src_lengths, dec_state)

                if len(model_output_tuple) == 5:
                    outputs, attns, dec_state, enc_output_dict, dec_output_dict = model_output_tuple
                elif len(model_output_tuple) == 3:
                    enc_output_dict = dec_output_dict = mem_dict = None
                    outputs, attns, dec_state = model_output_tuple

                target = batch.tgt[1:batch.tgt.size(0)]
                scores = self.model.decoder.generator(outputs.view(-1, outputs.size(2)))
                gtruth = target.view(-1)
                loss = F.cross_entropy(scores, gtruth, weight=self.weight, size_average=False) # , ignore_index=-100, reduce=None, reduction='elementwise_mean')
                loss_data = loss.data.clone()
                # batch_stats = self._stats(loss_data, scores.data, target.view(-1).data)
                batch_stats = self._stats(loss_data, F.log_softmax(scores.data), target.view(-1).data)

                loss.div(batch_size).backward()
                self.optim.step()

                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

            if report_func is not None:
                report_stats = report_func(self, epoch, i, len(self.train_iter), total_stats.start_time, self.optim.lr,
                                           report_stats, enc_output_dict, dec_output_dict, mem_dict)

        return total_stats

    def validate(self):
        """ Validate model.

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        with torch.no_grad():  # Don't store activations during evaluation
            for batch in self.valid_iter:
                src = onmt.io.make_features(batch, 'src', self.data_type)
                _, src_lengths = batch.src
                tgt = onmt.io.make_features(batch, 'tgt')

                target = batch.tgt[1:batch.tgt.size(0)]

                if self.opt.encoder_model == 'Rev' and self.opt.decoder_model == 'Rev':
                    tgt_lengths = torch.tensor([tgt.size(0)], device=tgt.device)
                    loss, num_words, num_correct, attn, enc_output_dict, dec_output_dict, mem_dict = self.model(src,
                                                                                                                tgt,
                                                                                                                src_lengths,
                                                                                                                tgt_lengths,
                                                                                                                self.weight,
                                                                                                                self.padding_idx)
                    loss_data = loss.data.clone()

                    batch_stats = onmt.Statistics(loss.item(), float(num_words), float(num_correct))
                    # batch_stats = self._stats(loss_data, scores.data, target.view(-1).data)

                    # Update statistics.
                    stats.update(batch_stats)
                else:
                    # F-prop through the model.
                    model_output_tuple = self.model(src, tgt, src_lengths)
                    outputs, attns = model_output_tuple[:2]

                    scores = self.model.decoder.generator(outputs.view(-1, outputs.size(2)))
                    gtruth = target.view(-1)
                    loss = F.cross_entropy(scores, gtruth, weight=self.weight, size_average=False)
                    loss_data = loss.data.clone()
                    batch_stats = self._stats(loss_data, scores.data, target.view(-1).data)

                    # Update statistics.
                    stats.update(batch_stats)

        # Set model back to training mode.
        self.model.train()

        return stats


    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`Statistics` : statistics for this batch.
        """
        non_padding = target.ne(self.padding_idx)
        if scores is not None:
            pred = scores.max(1)[1]
            num_correct = pred.eq(target).masked_select(non_padding).sum()
        else:
            num_correct = 0
        return onmt.Statistics(loss.item(), float(non_padding.sum()), float(num_correct))


    def epoch_step(self, ppl, epoch):
        return self.optim.update_learning_rate(ppl, epoch)


    def drop_checkpoint(self, opt, epoch, fields, valid_stats):
        """ Save a resumable checkpoint.

        Args:
            opt (dict): option object
            epoch (int): epoch number
            fields (dict): fields and vocabulary
            valid_stats : statistics of last validation run
        """
        model_state_dict = self.model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'vocab': onmt.io.save_fields_to_vocab(fields),
            'opt': opt,
            'epoch': epoch,
            'optim': self.optim,
        }

        save_path = os.path.join(opt.save_dir, 'best_checkpoint.pt')
        torch.save(checkpoint, save_path)
        return save_path
