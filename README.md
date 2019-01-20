# Code for ["Reversible Recurrent Neural Networks" (NeurIPS 2018)](https://arxiv.org/abs/1810.10999)

This repository contains the code for the paper [Reversible Recurrent Neural Networks (NeurIPS 2018)](https://arxiv.org/abs/1810.10999).

For language modeling, we build on the [Salesforce AWD-LSTM codebase](https://github.com/salesforce/awd-lstm-lm); for NMT, we use [OpenNMT](https://github.com/OpenNMT/OpenNMT).

# Requirements

* Python 3
* PyTorch 0.4.0
* torchtext 0.2.3 (for NMT)

The following is an example for how to set up a conda environment with appropriate versions of the packages:

```
conda create -n rev-rnn-env python=3.6
source activate rev-rnn-env
conda install pytorch=0.4.0 torchvision cuda80 -c pytorch
pip install -r requirements.txt
```

# Experiments

## Language Modeling

### PTB

**Baseline models**
```
python train.py --model lstm
```
```
python train.py --model gru
```

**Reversible models**

Use ```--use_buffers``` to track memory savings without actually training with reversibility, which is slower.
```
python train.py --model revgru --use_buffers
```
Use ```--use_reverse``` to train the model by reversing it.
```
python train.py --model revgru --use_reverse
```

### WikiText2

Use ```--data wt2``` to train on the WikiText2 dataset.

```
python train.py --data wt2
```

## Neural Machine Translation

### Multi30k

#### Data Setup

To run the Multi30k experiments, you first need to download and pre-process the Multi30k data:

```
./download_multi30k.sh
```


#### Training Baseline and Reversible Models

**Baseline LSTM**
```
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Vanilla -decoder_model=Vanilla -encoder_rnn_type=lstm -decoder_rnn_type=lstm -learning_rate=1 -context_type=hidden -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4
```


**RevLSTM Emb + 20H, with max 3 bits forgetting**
```
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revlstm -decoder_rnn_type=revlstm -learning_rate=0.5 -context_type=slice_emb -slice_dim=20 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse
```


**Baseline GRU**
```
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Vanilla -decoder_model=Vanilla -encoder_rnn_type=gru -decoder_rnn_type=gru -learning_rate=0.2 -context_type=hidden -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4
```


**RevGRU Emb + 20H, with max 3 bits forgetting**
```
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice_emb -slice_dim=20 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse
```

Complete lists of training commands needed to run the experiments in the paper are provided in `nmt/train_script/train_script_multi30k_revlstm.sh` and `nmt/train_script/train_script_multi30k_revgru.sh`.



### IWSLT

#### Data Setup

To run the IWSLT experiments, you will first need to download and pre-process the IWSLT data:

```
./download_iwslt.sh
```


#### Training Baseline and Reversible Models

**Baseline LSTM**
```
python train.py -data data/en-de/en-de.tok.low -save_model iwslt-model -start_decay_at=100 -epochs=80 -word_vec_size=600 -rnn_size=600 -enc_layers=2 -dec_layers=2 -batch_size=64 -encoder_model=Vanilla -decoder_model=Vanilla -encoder_rnn_type=lstm -decoder_rnn_type=lstm -learning_rate=1 -context_type=hidden -global_attention=general -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -gpuid=0 -no_log_during_epoch=1
```


```
python translate.py -gpu 0 -src data/en-de/IWSLT16.TED.tst2014.en-de.en.tok -tgt data/en-de/IWSLT16.TED.tst2014.en-de.de.tok -replace_unk -output iwslt.test.pred.tok -model saves/2018-05-16-enc:Vanilla-dec:Vanilla-et:LSTM-dt:LSTM-h:600-el:2-dl:2-em:600-atn:general-cxt:hidden-sl:10/best_checkpoint.pt

perl tools/multi-bleu.perl data/en-de/IWSLT16.TED.tst2014.en-de.de.tok < iwslt.test.pred.tok
```



**RevLSTM Emb + 60H**
```
python train.py -data data/en-de/en-de.tok.low -save_model iwslt-model -start_decay_at=100 -epochs=80 -word_vec_size=600 -rnn_size=600 -enc_layers=2 -dec_layers=2 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revlstm -decoder_rnn_type=revlstm -learning_rate=0.5 -context_type=slice_emb -slice_dim=60 -global_attention=general -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -dropouts=0.4 -gpuid=0 -no_log_during_epoch=1 -use_buffers -separate_buffers -use_reverse
```


```
python translate.py -gpu 0 -src data/en-de/IWSLT16.TED.tst2014.en-de.en.tok -tgt data/en-de/IWSLT16.TED.tst2014.en-de.de.tok -replace_unk -output iwslt.test.pred.tok -model saves/2018-05-17-enc:Rev-dec:Rev-et:RevLSTM5-dt:RevLSTM5-h:600-el:2-dl:2-em:600-atn:general-cxt:slice_emb-sl:60-ef1:0.875-ef2:0.875-df1:0.875-df2:0.875/best_checkpoint.pt

perl tools/multi-bleu.perl data/en-de/IWSLT16.TED.tst2014.en-de.de.tok < iwslt.test.pred.tok
```


**Baseline GRU**
```
python train.py -data data/en-de/en-de.tok.low -save_model iwslt-model -start_decay_at=100 -epochs=80 -word_vec_size=600 -rnn_size=600 -enc_layers=2 -dec_layers=2 -batch_size=64 -encoder_model=Vanilla -decoder_model=Vanilla -encoder_rnn_type=gru -decoder_rnn_type=gru -learning_rate=0.2 -context_type=hidden -global_attention=general -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -gpuid=0 -no_log_during_epoch=1
```


```
python translate.py -gpu 0 -src data/en-de/IWSLT16.TED.tst2014.en-de.en.tok -tgt data/en-de/IWSLT16.TED.tst2014.en-de.de.tok -replace_unk -output iwslt.test.pred.tok -model saves/2018-05-17-enc:Vanilla-dec:Vanilla-et:GRU-dt:GRU-h:600-el:2-dl:2-em:600-atn:general-cxt:hidden-sl:10/best_checkpoint.pt

perl tools/multi-bleu.perl data/en-de/IWSLT16.TED.tst2014.en-de.de.tok < iwslt.test.pred.tok
```


**RevGRU Emb + 60H**
```
python train.py -data data/en-de/en-de.tok.low -save_model iwslt-model -start_decay_at=100 -epochs=80 -word_vec_size=600 -rnn_size=600 -enc_layers=2 -dec_layers=2 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice_emb -slice_dim=60 -global_attention=general -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -dropouts=0.4 -gpuid=0 -no_log_during_epoch=1 -use_buffers -separate_buffers -use_reverse
```


```
python translate.py -gpu 0 -src data/en-de/IWSLT16.TED.tst2014.en-de.en.tok -tgt data/en-de/IWSLT16.TED.tst2014.en-de.de.tok -replace_unk -output iwslt.test.pred.tok -model saves/2018-05-17-enc:Rev-dec:Rev-et:RevGRU-dt:RevGRU-h:600-el:2-dl:2-em:600-atn:general-cxt:slice_emb-sl:60-ef1:0.875-ef2:0.875-df1:0.875-df2:0.875/best_checkpoint.pt

perl tools/multi-bleu.perl data/en-de/IWSLT16.TED.tst2014.en-de.de.tok < iwslt.test.pred.tok
```


The training commands needed to run the IWSLT experiment are provided in `nmt/train_scripts/train_script_iwslt_revlstm.sh` and `nmt/train_script/train_script_iwslt_revgru.sh`.


# Citation

If you use this code, please cite:

```
@inproceedings{revrnn2018,
  title={Reversible Recurrent Neural Networks},
  author={Matthew MacKay and Paul Vicol and Jimmy Ba and Roger Grosse},
  booktitle={{Neural Information Processing Systems (NeurIPS)}},
  year={2018}
}
```
