# =================================================================================================
# BASELINE GRU
# =================================================================================================

# Vanilla GRU Baseline Full Hidden State
# --------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Vanilla -decoder_model=Vanilla -encoder_rnn_type=gru -decoder_rnn_type=gru -learning_rate=0.2 -context_type=hidden -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4


# =================================================================================================
# MAX FORGETTING 1 BIT
# =================================================================================================

# RevGRU h[:20] dim  Max forget 1 bit
# -----------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice -slice_dim=20 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=0.5 -dec_max_forget=0.5 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU h[:100] dim  Max forget 1 bit
# ------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice -slice_dim=100 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=0.5 -dec_max_forget=0.5 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU Whole Hidden  Max forget 1 bit
# -------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice -slice_dim=300 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=0.5 -dec_max_forget=0.5 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU Embeddings  Max forget 1 bit
# -----------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=emb -slice_dim=0 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=0.5 -dec_max_forget=0.5 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU [emb; h[:20]] dim  Max forget 1 bit
# ------------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice_emb -slice_dim=20 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=0.5 -dec_max_forget=0.5 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse




# =================================================================================================
# MAX FORGETTING 2 BITS
# =================================================================================================

# RevGRU h[:20] dim  Max forget 2 bits
# ------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice -slice_dim=20 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=0.75 -dec_max_forget=0.75 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU h[:100] dim  Max forget 2 bits
# -------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice -slice_dim=100 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=0.75 -dec_max_forget=0.75 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU Whole Hidden  Max forget 2 bits
# --------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice -slice_dim=300 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=0.75 -dec_max_forget=0.75 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU Embeddings  Max forget 2 bits
# ------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=emb -slice_dim=0 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=0.75 -dec_max_forget=0.75 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU [emb; h[:20]] dim  Max forget 2 bits
# -------------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice_emb -slice_dim=20 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=0.75 -dec_max_forget=0.75 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse




# =================================================================================================
# MAX FORGETTING 3 BITS
# =================================================================================================

# RevGRU h[:20] dim  Max forget 3 bits
# ------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice -slice_dim=20 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU h[:100] dim  Max forget 3 bits
# -------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice -slice_dim=100 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU Whole Hidden  Max forget 3 bits
# --------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice -slice_dim=300 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU Embeddings  Max forget 3 bits
# ------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=emb -slice_dim=0 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU [emb; h[:20]] dim  Max forget 3 bits
# -------------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice_emb -slice_dim=20 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse




# =================================================================================================
# MAX FORGETTING 5 BITS
# =================================================================================================

# RevGRU h[:20] dim  Max forget 5 bits
# ------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice -slice_dim=20 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=0.96875 -dec_max_forget=0.96875 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU h[:100] dim  Max forget 5 bits
# -------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice -slice_dim=100 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=0.96875 -dec_max_forget=0.96875 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU Whole Hidden  Max forget 5 bits
# --------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice -slice_dim=300 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=0.96875 -dec_max_forget=0.96875 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU Embeddings  Max forget 5 bits
# ------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=emb -slice_dim=0 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=0.96875 -dec_max_forget=0.96875 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU [emb; h[:20]] dim  Max forget 5 bits
# -------------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice_emb -slice_dim=20 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=0.96875 -dec_max_forget=0.96875 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse




# =================================================================================================
# MAX FORGETTING NO LIMIT
# =================================================================================================

# RevGRU h[:20] dim  Max forget no limit
# --------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice -slice_dim=20 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=1 -dec_max_forget=1 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU h[:100] dim  Max forget no limit
# ---------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice -slice_dim=100 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=1 -dec_max_forget=1 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU Whole Hidden  Max forget no limit
# ----------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice -slice_dim=300 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=1 -dec_max_forget=1 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU Embeddings  Max forget no limit
# --------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=emb -slice_dim=0 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=1 -dec_max_forget=1 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse


# RevGRU [emb; h[:20]] dim  Max forget no limit
# ---------------------------------------------
python train.py -data data/multi30k/multi30k.tok.low -save_model multi30k-model -start_decay_at=100 -epochs=80 -word_vec_size=300 -rnn_size=300 -enc_layers=1 -dec_layers=1 -batch_size=64 -encoder_model=Rev -decoder_model=Rev -encoder_rnn_type=revgru -decoder_rnn_type=revgru -learning_rate=0.2 -context_type=slice_emb -slice_dim=20 -global_attention=general -gpuid=0 -dropouti=0.4 -dropouto=0.4 -dropouth=0.4 -enc_max_forget=1 -dec_max_forget=1 -dropouts=0.4 -use_buffers -separate_buffers -use_reverse
