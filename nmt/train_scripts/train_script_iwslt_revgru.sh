# Vanilla GRU Baseline Full Hidden State
# ======================================

# Train
# -----
python train.py -data data/en-de/en-de.tok.low -save_model iwslt-model -start_decay_at=100 -epochs=80 -word_vec_size=600 -rnn_size=600 -enc_layers=2 -dec_layers=2 -batch_size=64 -encoder_model=Vanilla -decoder_model=Vanilla -encoder_rnn_type=gru -decoder_rnn_type=gru -learning_rate=0.2 -context_type=hidden -global_attention=general -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -gpuid=0 -no_log_during_epoch=1


# Evaluate
# --------
python translate.py -gpu 0 -src data/en-de/IWSLT16.TED.tst2014.en-de.en.tok -tgt data/en-de/IWSLT16.TED.tst2014.en-de.de.tok -replace_unk -output iwslt.test.pred.tok -model saves/2018-05-17-enc:Vanilla-dec:Vanilla-et:GRU-dt:GRU-h:600-el:2-dl:2-em:600-atn:general-cxt:hidden-sl:10/best_checkpoint.pt

perl tools/multi-bleu.perl data/en-de/IWSLT16.TED.tst2014.en-de.de.tok < iwslt.test.pred.tok



# RevGRU [emb; h[:60]]  Max forgetting 3 bits
# ===========================================

# Train
# -----
python train.py -data data/en-de/en-de.tok.low -save_model iwslt-model -start_decay_at=100 -epochs=80 -word_vec_size=600 -rnn_size=600 -enc_layers=2 -dec_layers=2 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice_emb -slice_dim=60 -global_attention=general -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -dropouts=0.4 -gpuid=0 -no_log_during_epoch=1 -use_buffers -separate_buffers -use_reverse


# Evaluate
# --------
python translate.py -gpu 0 -src data/en-de/IWSLT16.TED.tst2014.en-de.en.tok -tgt data/en-de/IWSLT16.TED.tst2014.en-de.de.tok -replace_unk -output iwslt.test.pred.tok -model saves/2018-05-17-enc:Rev-dec:Rev-et:RevGRU-dt:RevGRU-h:600-el:2-dl:2-em:600-atn:general-cxt:slice_emb-sl:60-ef1:0.875-ef2:0.875-df1:0.875-df2:0.875/best_checkpoint.pt

perl tools/multi-bleu.perl data/en-de/IWSLT16.TED.tst2014.en-de.de.tok < iwslt.test.pred.tok
