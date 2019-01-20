#!/usr/bin/env python
import os
import re
import sys
import ipdb
import glob
import random
import argparse
import datetime
import subprocess

# YAML setup
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']

import torch
import torch.nn as nn
from torch import cuda

import opts
import onmt
import onmt.io
import onmt.Models
import onmt.modules
import onmt.ModelConstructor
from onmt.Utils import use_gpu


parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)

opt = parser.parse_args()

if opt.word_vec_size != -1:
    opt.src_word_vec_size = opt.word_vec_size
    opt.tgt_word_vec_size = opt.word_vec_size

if opt.layers != -1:
    opt.enc_layers = opt.layers
    opt.dec_layers = opt.layers

opt.brnn = (opt.encoder_type == "brnn")
if opt.seed > 0:
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.gpuid:
    print("WARNING: You have a CUDA device, should run with -gpuid 0")

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)


def report_func(trnr, epoch, batch, num_batches, start_time, lr, report_stats, enc_output_dict, dec_output_dict, mem_dict):
    """
    This is the user-defined batch-level traing progress
    report function.

    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    global iteration_log_loss_file
    global mem_log_file

    if batch % opt.report_every == -1 % opt.report_every:
        report_stats.output(epoch, batch+1, num_batches, start_time)

        if mem_dict:
            if opt.separate_buffers:
                enc_used_bits = mem_dict['enc_used_bits']
                dec_used_bits = mem_dict['dec_used_bits']
                enc_optimal_bits = mem_dict['enc_optimal_bits']
                dec_optimal_bits = mem_dict['dec_optimal_bits']
                enc_normal_bits = mem_dict['enc_normal_bits']
                dec_normal_bits = mem_dict['dec_normal_bits']

                enc_actual_ratio = enc_normal_bits / enc_used_bits
                enc_optimal_ratio = enc_normal_bits / enc_optimal_bits
                dec_actual_ratio = dec_normal_bits / dec_used_bits
                dec_optimal_ratio = dec_normal_bits / dec_optimal_bits

                print("Enc actual memory ratio {}".format(enc_actual_ratio))
                print("Enc optimal memory ratio {}".format(enc_optimal_ratio))
                print("Dec actual memory ratio {}".format(dec_actual_ratio))
                print("Dec optimal memory ratio {}".format(dec_optimal_ratio))

                mem_log_file.write('{} {} {} {} {} {}\n'.format(
                                    epoch, batch, enc_actual_ratio, enc_optimal_ratio, dec_actual_ratio, dec_optimal_ratio))
                mem_log_file.flush()
            else:
                used_bits = mem_dict['used_bits']
                optimal_bits = mem_dict['optimal_bits']
                normal_bits = mem_dict['normal_bits']

                actual_ratio = normal_bits / used_bits
                optimal_ratio = normal_bits / optimal_bits

                print("Actual memory ratio {}".format(actual_ratio))
                print("Optimal memory ratio {}".format(optimal_ratio))

                mem_log_file.write('{} {} {} {}\n'.format(
                                    epoch, batch, actual_ratio, optimal_ratio))
                mem_log_file.flush()


        if not opt.no_log_during_epoch:
            valid_stats = trnr.validate()

            iteration_log_loss_file.write('{} {} {} {} {} {}\n'.format(
                                           epoch, batch, report_stats.accuracy(), report_stats.ppl(), valid_stats.accuracy(),
                                           valid_stats.ppl()))
            iteration_log_loss_file.flush()

        report_stats = onmt.Statistics()

    return report_stats


def make_train_data_iter(train_dataset, opt):
    """
    This returns user-defined train data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    # Sort batch by decreasing lengths of sentence required by pytorch.
    # sort=False means "Use dataset's sortkey instead of iterator's".
    return onmt.io.OrderedIterator(
                dataset=train_dataset, batch_size=opt.batch_size,
                device=opt.gpuid[0] if opt.gpuid else -1,
                sort=False, sort_within_batch=True, repeat=False)


def make_valid_data_iter(valid_dataset, opt):
    """
    This returns user-defined validate data iterator for the trainer
    to iterate over during each validate epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    is ok too.
    """
    # Sort batch by decreasing lengths of sentence required by pytorch.
    # sort=False means "Use dataset's sortkey instead of iterator's".
    return onmt.io.OrderedIterator(
                dataset=valid_dataset, batch_size=opt.valid_batch_size,
                device=opt.gpuid[0] if opt.gpuid else -1,
                train=False, sort=False, sort_within_batch=True)


iteration_log_loss_file = None
mem_log_file = None


def train_model(model, train_dataset, valid_dataset, fields, optim, model_opt):

    global iteration_log_loss_file
    global mem_log_file

    train_iter = make_train_data_iter(train_dataset, opt)
    valid_iter = make_valid_data_iter(valid_dataset, opt)

    trunc_size = opt.truncated_decoder
    shard_size = opt.max_generator_batches
    data_type = train_dataset.data_type

    trainer = onmt.Trainer(model,
                           train_iter,
                           valid_iter,
                           fields["tgt"].vocab,
                           optim,
                           trunc_size,
                           shard_size,
                           data_type,
                           opt=model_opt)

    log_perp = open(os.path.join(opt.save_dir, 'log_perp'), 'w')
    iteration_log_loss_file = open(os.path.join(opt.save_dir, 'iteration_log_loss'), 'w')
    mem_log_file = open(os.path.join(opt.save_dir, 'mem_log'), 'w')

    best_val_ppl = 1e6
    best_val_checkpoint_path = None

    # You can press Ctrl+C at any time to exit training early, and print the best model checkpoint.
    try:
        for epoch in range(opt.start_epoch, opt.epochs + 1):
            print('')

            # 1. Train for one epoch on the training set.
            train_stats = trainer.train(epoch, report_func)
            print('Train perplexity: %g' % train_stats.ppl())
            print('Train accuracy: %g' % train_stats.accuracy())

            # 2. Validate on the validation set.
            valid_stats = trainer.validate()
            print('Validation perplexity: %g' % valid_stats.ppl())
            print('Validation accuracy: %g' % valid_stats.accuracy())

            # 3. Logging
            # Log to remote server.
            if opt.exp_host:
                train_stats.log("train", experiment, optim.lr)
                valid_stats.log("valid", experiment, optim.lr)

            # Write train and val perplexities to a file
            log_perp.write("{} {} {}\n".format(epoch, train_stats.ppl(), valid_stats.ppl()))
            log_perp.flush()

            # 4. Update the learning rate
            trainer.epoch_step(valid_stats.ppl(), epoch)

            # 5. Drop a checkpoint if needed.
            if epoch >= opt.start_checkpoint_at:
                checkpoint_path = trainer.drop_checkpoint(model_opt, epoch, fields, valid_stats)

                if valid_stats.ppl() < best_val_ppl:
                    best_val_ppl = valid_stats.ppl()
                    best_val_checkpoint_path = checkpoint_path
    except KeyboardInterrupt:
        print("Exiting from training early!")

    print('\nBest val checkpoint: {}'.format(best_val_checkpoint_path))

    return best_val_checkpoint_path


def extract_bleu_score(output_string):
    output_string = output_string.strip().split('\n')[-1]  # Gets last line of output, like ">> BLEU = 9.79, 37.2/13.3/6.6/3.1 (BP=0.978, ratio=0.978, hyp_len=11976, ref_len=12242)"
    output_string = output_string.split(',')[0]    # Gets ">> BLEU = 9.79"
    m = re.search('>> BLEU = (.+)', output_string)
    return m.group(1)


def evaluate(best_val_checkpoint_path):
    # python translate.py -src data/multi30k/test2016.en.atok -output pred.txt \
    #                     -replace_unk -tgt=data/multi30k/test2016.de.atok -report_bleu -gpu 2
    #                     -model saves/2018-02-09-enc:Rev-dec:Rev-et:RevGRU-dt:RevGRU-h:300-el:1-dl:1-em:300-atn:general-cxt:slice_emb-sl:20-ef1:0.875-ef2:0.875-df1:0.875-df2:0.875/best_checkpoint.pt

    base_dir = os.path.dirname(best_val_checkpoint_path)

    if '600' in best_val_checkpoint_path:
        test_output = subprocess.run(['python', 'translate.py', '-src', 'data/en-de/IWSLT16.TED.tst2014.en-de.en.tok.low',
                                      '-output', os.path.join(base_dir, 'test_pred.txt'), '-replace_unk', '-tgt', 'data/en-de/IWSLT16.TED.tst2014.en-de.de.tok.low',
                                      '-report_bleu', '-gpu', str(opt.gpuid[0]), '-model', best_val_checkpoint_path], stdout=subprocess.PIPE)

        test_output_string = test_output.stdout.decode('utf-8')
        print(test_output_string)

        # Also save the whole stdout string for reference
        with open(os.path.join(base_dir, 'test_stdout.txt'), 'w') as f:
            f.write('{}\n'.format(test_output_string))

        val_output = subprocess.run(['python', 'translate.py', '-src', 'data/en-de/IWSLT16.TED.tst2013.en-de.en.tok.low',
                                     '-output', os.path.join(base_dir, 'val_pred.txt'), '-replace_unk', '-tgt', 'data/en-de/IWSLT16.TED.tst2013.en-de.de.tok.low',
                                     '-report_bleu', '-gpu', str(opt.gpuid[0]), '-model', best_val_checkpoint_path], stdout=subprocess.PIPE)

        val_output_string = val_output.stdout.decode('utf-8')
        print(val_output_string)
    else:
        test_output = subprocess.run(['python', 'translate.py', '-src', 'data/multi30k/test2016.en.tok.low',
                                      '-output', os.path.join(base_dir, 'test_pred.txt'), '-replace_unk', '-tgt', 'data/multi30k/test2016.de.tok.low',
                                      '-report_bleu', '-gpu', str(opt.gpuid[0]), '-model', best_val_checkpoint_path], stdout=subprocess.PIPE)

        test_output_string = test_output.stdout.decode('utf-8')
        print(test_output_string)

        # Also save the whole stdout string for reference
        with open(os.path.join(base_dir, 'test_stdout.txt'), 'w') as f:
            f.write('{}\n'.format(test_output_string))

        val_output = subprocess.run(['python', 'translate.py', '-src', 'data/multi30k/val.en.tok.low',
                                     '-output', os.path.join(base_dir, 'val_pred.txt'), '-replace_unk', '-tgt', 'data/multi30k/val.de.tok.low',
                                     '-report_bleu', '-gpu', str(opt.gpuid[0]), '-model', best_val_checkpoint_path], stdout=subprocess.PIPE)

        val_output_string = val_output.stdout.decode('utf-8')
        print(val_output_string)

    # Also save the whole stdout string for reference
    with open(os.path.join(base_dir, 'val_stdout.txt'), 'w') as f:
        f.write('{}\n'.format(val_output_string))

    val_bleu = extract_bleu_score(val_output_string)
    test_bleu = extract_bleu_score(test_output_string)

    with open(os.path.join(base_dir, 'result.txt'), 'w') as f:
        f.write('{} {}\n'.format(val_bleu, test_bleu))

    print('Val BLEU: {} | Test BLEU: {}'.format(val_bleu, test_bleu))


def check_save_model_path():
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def load_dataset(data_type):
    assert data_type in ["train", "valid"]

    print("Loading %s data from '%s'" % (data_type, opt.data))

    pts = glob.glob(opt.data + '.' + data_type + '.[0-9]*.pt')
    if pts:
        # Multiple onmt.io.*Dataset's, coalesce all.
        # torch.load loads them imemediately, which might eat up
        # too much memory. A lazy load would be better, but later
        # when we create data iterator, it still requires these
        # data to be loaded. So it seams we don't have a good way
        # to avoid this now.
        datasets = []
        for pt in pts:
            datasets.append(torch.load(pt))
        dataset = onmt.io.ONMTDatasetBase.coalesce_datasets(datasets)
    else:
        # Only one onmt.io.*Dataset, simple!
        dataset = torch.load(opt.data + '.' + data_type + '.pt')

    print(' * number of %s sentences: %d' % (data_type, len(dataset)))

    return dataset


def load_fields(train_dataset, valid_dataset, checkpoint):
    data_type = train_dataset.data_type

    fields = onmt.io.load_fields_from_vocab(torch.load(opt.data + '.vocab.pt'), data_type)
    fields = dict([(k, f) for (k, f) in fields.items() if k in train_dataset.examples[0].__dict__])

    # We save fields in vocab.pt, so assign them back to dataset here.
    train_dataset.fields = fields
    valid_dataset.fields = fields

    if opt.train_from:
        print('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = onmt.io.load_fields_from_vocab(checkpoint['vocab'], data_type)

    if data_type == 'text':
        print(' * vocabulary size. source = %d; target = %d' %
              (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    else:
        print(' * vocabulary size. target = %d' %
              (len(fields['tgt'].vocab)))

    return fields


def collect_report_features(fields):
    src_features = onmt.io.collect_features(fields, side='src')
    tgt_features = onmt.io.collect_features(fields, side='tgt')

    for j, feat in enumerate(src_features):
        print(' * src feature %d size = %d' % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        print(' * tgt feature %d size = %d' % (j, len(fields[feat].vocab)))


def build_model(model_opt, opt, fields, checkpoint):
    print('Building model...')
    model = onmt.ModelConstructor.make_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    if len(opt.gpuid) > 1:
        print('Multi gpu training: ', opt.gpuid)
        model = nn.DataParallel(model, device_ids=opt.gpuid, dim=1)
    print(model)
    return model


def build_optim(model, checkpoint):
    if opt.train_from:
        print('Loading optimizer from checkpoint.')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
    else:
        optim = onmt.Optim(opt.optim,  # SGD by default
                           opt.learning_rate,
                           opt.max_grad_norm,
                           lr_decay=opt.learning_rate_decay,
                           start_decay_at=opt.start_decay_at,
                           beta1=opt.adam_beta1,
                           beta2=opt.adam_beta2,
                           adagrad_accum=opt.adagrad_accumulator_init,
                           decay_method=opt.decay_method,
                           warmup_steps=opt.warmup_steps,
                           model_size=opt.rnn_size)

    optim.set_parameters(model.parameters())

    return optim


def main():

    #######################################################################
    ### Create save folder
    ### Example: saves/2018-01-21-enc:CUDNN-dec:CUDNN-etype:LSTM-dtype:LSTM
    #######################################################################
    timestamp = '{:%Y-%m-%d}'.format(datetime.datetime.now())

    if opt.encoder_model == 'Rev' or opt.decoder_model == 'Rev':
        exp_name = '{}-enc:{}-dec:{}-et:{}-dt:{}-h:{}-el:{}-dl:{}-em:{}-atn:{}-cxt:{}-sl:{}-ef:{}-df:{}-di:{}-dh:{}-do:{}-ds:{}-lr:{}-init:{}-userev:{}'.format(
                    timestamp, opt.encoder_model, opt.decoder_model, opt.encoder_rnn_type, opt.decoder_rnn_type,
                    opt.rnn_size, opt.enc_layers, opt.dec_layers, opt.word_vec_size,
                    opt.global_attention, opt.context_type, opt.slice_dim,
                    opt.enc_max_forget, opt.dec_max_forget, opt.dropouti, opt.dropouth, opt.dropouto, opt.dropouts, opt.learning_rate, opt.param_init, int(opt.use_reverse))
    else:
        exp_name = '{}-enc:{}-dec:{}-et:{}-dt:{}-h:{}-el:{}-dl:{}-em:{}-atn:{}-cxt:{}-sl:{}-di:{}-dh:{}-do:{}-ds:{}-lr:{}-init:{}-userev:{}'.format(
                    timestamp, opt.encoder_model, opt.decoder_model, opt.encoder_rnn_type, opt.decoder_rnn_type,
                    opt.rnn_size, opt.enc_layers, opt.dec_layers, opt.word_vec_size,
                    opt.global_attention, opt.context_type, opt.slice_dim, opt.dropouti, opt.dropouth, opt.dropouto, opt.dropouts, opt.learning_rate, opt.param_init, int(opt.use_reverse))

    opt.save_dir = os.path.join(opt.save_dir, exp_name)

    if os.path.exists(os.path.join(opt.save_dir, 'result.txt')):
        print('The result file {} exists! Terminating to not overwrite it!'.format(os.path.join(opt.save_dir, 'result.txt')))
        sys.exit(0)

    # Create save dir if it doesn't exist
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    # Save command-line arguments
    with open(os.path.join(opt.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(opt), f)

    # Load train and validate data.
    train_dataset = load_dataset("train")
    valid_dataset = load_dataset("valid")
    print(' * maximum batch size: %d' % opt.batch_size)

    # Load checkpoint if resuming from a previous training run.
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from, map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        opt.start_epoch = checkpoint['epoch'] + 1
    else:
        checkpoint = None
        model_opt = opt

    # Load fields generated from preprocess phase.
    fields = load_fields(train_dataset, valid_dataset, checkpoint)

    # Report src/tgt features.
    collect_report_features(fields)

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    tally_parameters(model)

    # Build optimizer.
    optim = build_optim(model, checkpoint)

    # Do training.
    best_val_checkpoint_path = train_model(model, train_dataset, valid_dataset, fields, optim, model_opt)

    # Evaluate the final model on the validation and test sets
    evaluate(best_val_checkpoint_path)


if __name__ == "__main__":
    main()
