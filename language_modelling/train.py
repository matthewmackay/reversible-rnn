import argparse
import time
import math
import torch
import torch.nn as nn
import numpy as np
import baselines.model as base_model
import rev_training.model as rev_model
import usual_training.model as usual_model
from datetime import datetime

import data

# --gres=gpu:1 --job-name=e0_2 --exclude=dgx1,guppy9,guppy15,guppy18,guppy22,guppy27,guppy30,guppy28,guppy29,guppy31,guppy33,guppy36 -c 2 -l -p gpuc python train.py 
###############################################################################
# Hyperparameters
###############################################################################
datasets = ['ptb', 'wt2']
models = ['gru', 'lstm', 'revgru', 'revlstm']
parser = argparse.ArgumentParser(description='PyTorch PennTreeBank Reversible Language Model (Usual Training)')
parser.add_argument('--data', type=str, default='ptb', help='which dataset to use', choices=datasets)
parser.add_argument('--model', type=str, default='revgru', help='type of recurrent net', choices=models)
# Model capacity hyperparameters.
parser.add_argument('--emb_size', type=int, default=650, help='size of word embeddings')
parser.add_argument('--h_size', type=int, default=1150, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1, metavar='N', help='number of layers to bits-GRU')
parser.add_argument('--max_forget', type=float, default=0.875, help='max no. of bits forgotten per hidden unit per time step')
# Optimization hyperparameters.
parser.add_argument('--lr', type=float, default=20., help='initial learning rate')
parser.add_argument('--lr_decay', type=float, default=4., help='how much to decay learning rate by when triggered')
parser.add_argument('--clip', type=float, default=0.1, help='gradient clipping')
parser.add_argument('--bptt', type=int, default=70, help='sequence length')
parser.add_argument('--epochs', type=int, default=10000, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
parser.add_argument('--nonmono', type=int, default=5, help='how many epochs to wait for validation loss to decrease')
# Regularization hyperparameters
parser.add_argument('--dropout', type=float, default=0.4, help='dropout applied to layers fed into classifier (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3, help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.4, help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1, help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5, help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='weight decay applied to all weights')
# Fixed point/buffer hyperparameters.
parser.add_argument('--use_buffers', action='store_true', help='whether or not to use buffers (needed for exact reversibility)')
parser.add_argument('--use_reverse', action='store_true', help='whether or not to use reversible training')
# Miscellaneous hyperparameters.
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--no_cuda', action='store_true', help='disables CUDA training')
parser.add_argument('--log_interval', type=int, default=200, metavar='N', help='report interval')
parser.add_argument('--save', action='store_true', help='whether or not to save current run') 
parser.add_argument('--save_dir', type=str, default='frac-forget', help='which directory to save experiment in')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
if args.cuda: 
    torch.cuda.manual_seed(args.seed)

if args.model in ['gru', 'lstm']:
    model = base_model
elif args.model in ['revgru', 'revlstm']:
    if args.use_reverse:
        model = rev_model
    else:
        model = usual_model

###############################################################################
# Load data
###############################################################################
def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

def get_batch(source, i, args, seq_len=None):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

if args.data == 'ptb':
    data_dir = 'ptb_data'
if args.data == 'wt2':
    data_dir = 'wt2_data'
corpus = data.Corpus(data_dir)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args) 
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################
ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emb_size, args.h_size, args.nlayers,
    args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.max_forget,
    args.use_buffers)
model = model.to(device)
criterion = nn.CrossEntropyLoss() 

###############################################################################
# Training code
###############################################################################
def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hiddens(batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args)
            output_dict = model(data, hidden)
            output_flat = output_dict['decoded'].view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).data
            hidden = repackage_hidden(output_dict['last_h'])
    return total_loss.item() / len(data_source)

def train(global_step):
    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hiddens(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        # Adjust learning rate based on sequence size. We take larger steps for longer sequences.
        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        if args.use_reverse:
            targets = targets.view(-1, args.batch_size)
            output_dict = model.forward_and_backward(data, targets, hidden)
            loss = output_dict['loss']
            hidden = output_dict['last_h']
        else:
            output_dict = model(data, hidden)
            output = output_dict['decoded']
            hidden = output_dict['last_h']
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data

        # Change learning rate back to default after processing sequence fragment. 
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            # Compute memory saving ratios.
            if args.model in ['revgru', 'revlstm']:
                actual_ratio = float(output_dict['normal_bits']) / float(output_dict.get('used_bits', -1)) 
                optimal_ratio = float(output_dict['normal_bits']) / float(output_dict['optimal_bits'])
            else:
                actual_ratio = optimal_ratio = -1.
            
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | actual_ratio {:4.2f} | optimal_ratio {:4.2f}'
                    .format(epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), actual_ratio,
                    optimal_ratio))
            total_loss = 0
            start_time = time.time()

        ###
        batch += 1
        global_step += 1
        i += seq_len

    return global_step

# Loop over epochs.
lr = args.lr
best_val_loss = []
global_step = 0 # Keeps track of number of batches trained on.


# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        global_step = train(global_step)
        val_loss = evaluate(val_data, eval_batch_size)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)

        if (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
            print('Decaying learning rate!')
            optimizer.param_groups[0]['lr'] /= args.lr_decay

        if optimizer.param_groups[0]['lr'] < 1e-2:
            break
        
        best_val_loss.append(val_loss)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)