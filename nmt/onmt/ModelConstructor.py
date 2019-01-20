"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import ipdb

import torch
import torch.nn as nn

import onmt
import onmt.io
import onmt.Models
import onmt.modules
from onmt.Utils import use_gpu
from onmt.Models import NMTModel, RevNMTModel, RNNEncoder, StdRNNDecoder

import revencoder
import revdecoder
import custom_models


def make_embeddings(opt, word_dict, feature_dicts, for_encoder=True):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    if for_encoder:
        embedding_dim = opt.src_word_vec_size  # Equal to opt.word_vec_size, because we set it that way at the start of train.py
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[onmt.io.PAD_WORD]
    num_word_embeddings = len(word_dict)

    return nn.Embedding(num_word_embeddings, embedding_dim, padding_idx=word_padding_idx)


def make_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    if opt.encoder_model == 'Vanilla':
        return custom_models.MyEncoder(rnn_type=opt.encoder_rnn_type,
                                       nhid=opt.rnn_size,
                                       num_layers=opt.enc_layers,
                                       embeddings=embeddings,
                                       context_type=opt.context_type,
                                       slice_dim=opt.slice_dim,
                                       dropoute=opt.dropoute,
                                       dropouti=opt.dropouti,
                                       dropouth=opt.dropouth,
                                       dropouto=opt.dropouto,
                                       wdrop=opt.wdrop)
    elif opt.encoder_model == 'Rev':
        return revencoder.RevEncoder(rnn_type=opt.encoder_rnn_type,
                                     h_size=opt.rnn_size,
                                     nlayers=opt.enc_layers,
                                     embedding=embeddings,
                                     slice_dim=opt.slice_dim,
                                     max_forget=opt.enc_max_forget,
                                     use_buffers=opt.use_buffers,
                                     dropouti=opt.dropouti,
                                     dropouth=opt.dropouth,
                                     wdrop=opt.wdrop,
                                     context_type=opt.context_type)


def make_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    if opt.input_feed:
        return InputFeedRNNDecoder(opt.decoder_rnn_type, opt.brnn,
                                   opt.dec_layers, opt.rnn_size,
                                   opt.global_attention,
                                   opt.coverage_attn,
                                   opt.context_gate,
                                   opt.copy_attn,
                                   opt.dropout,
                                   embeddings)
    else:
        if opt.context_type == 'slice':
            context_size = opt.slice_dim
        elif opt.context_type == 'slice_emb':
            context_size = opt.slice_dim + opt.src_word_vec_size
        else:
            context_size = opt.rnn_size  # rnn_size is the size of the RNN hidden states, so this uses the full hidden state

        if opt.decoder_model == 'Vanilla':
            return custom_models.MyDecoder(opt.decoder_rnn_type,
                                           opt.dec_layers,
                                           opt.rnn_size,
                                           attn_type=opt.global_attention,  # opt.global_attention == 'general'
                                           context_size=context_size,
                                           dropout=opt.dropout,
                                           embeddings=embeddings,
                                           dropouti=opt.dropouti,
                                           dropouth=opt.dropouth,
                                           dropouto=opt.dropouto,
                                           wdrop=opt.wdrop)
        elif opt.decoder_model == 'Rev':
            return revdecoder.RevDecoder(opt.decoder_rnn_type,
                                         opt.rnn_size,
                                         opt.dec_layers,
                                         embeddings,
                                         attn_type=opt.global_attention,  # opt.global_attention == 'general'
                                         context_size=context_size,
                                         dropouti=opt.dropouti,
                                         dropouth=opt.dropouth,
                                         dropouto=opt.dropouto,
                                         wdrop=opt.wdrop,
                                         dropouts=opt.dropouts,
                                         max_forget=opt.dec_max_forget,
                                         context_type=opt.context_type,
                                         slice_dim=opt.slice_dim,
                                         use_buffers=opt.use_buffers)


def load_test_model(opt, dummy_opt):
    checkpoint = torch.load(opt.model, map_location=lambda storage, loc: storage)
    fields = onmt.io.load_fields_from_vocab(checkpoint['vocab'], data_type=opt.data_type)

    model_opt = checkpoint['opt']
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]

    model = make_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    model.eval()
    return fields, model, model_opt


def make_base_model(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """

    # Make encoder.
    src_dict = fields["src"].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(fields, 'src')
    src_embeddings = make_embeddings(model_opt, src_dict, feature_dicts)
    encoder = make_encoder(model_opt, src_embeddings)

    # Make decoder.
    tgt_dict = fields["tgt"].vocab
    feature_dicts = onmt.io.collect_feature_vocabs(fields, 'tgt')
    tgt_embeddings = make_embeddings(model_opt, tgt_dict, feature_dicts, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')

        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    decoder = make_decoder(model_opt, tgt_embeddings)

    # Make NMTModel(= encoder + decoder).
    if model_opt.decoder_model == 'Rev' and model_opt.encoder_model == 'Rev':
        model = RevNMTModel(encoder, decoder, fields['tgt'].vocab, opt=model_opt)
    else:
        model = NMTModel(encoder, decoder, opt=model_opt)

    model.model_type = model_opt.model_type

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        print('Loading model parameters.')
        model.load_state_dict(checkpoint['model'])
    else:
        if model_opt.param_init != 0.0:
            print('Intializing model parameters.')
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)

    # Make the whole model leverage GPU if indicated to do so.
    if gpu:
        model.cuda()
    else:
        model.cpu()

    return model
