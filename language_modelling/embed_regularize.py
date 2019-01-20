import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1))
        mask = mask.bernoulli_(1 - dropout)
        mask = mask.expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
    X = F.embedding(words, masked_embed_weight, padding_idx, embed.max_norm,
        embed.norm_type, embed.scale_grad_by_freq, embed.sparse)
    return X

class EmbedDropout(nn.Module):

    def __init__(self, embed, dropout):
        super(EmbedDropout, self).__init__()
        self.embed = embed
        self.dropout = dropout
        self._setup()

    def _setup(self):
        w = self.embed.weight
        del self.embed._parameters['weight']
        self.embed.register_parameter('raw_weight', nn.Parameter(w.data))

    def set_mask(self):
        mask = self.embed.raw_weight.data.new().resize_((self.embed.raw_weight.size(0), 1))
        mask = mask.bernoulli_(1 - self.dropout)
        self.mask = mask.expand_as(self.embed.raw_weight) / (1 - self.dropout)

    def set_weight(self):
        raw_w = self.embed.raw_weight.clone()
        if self.training:
            w = self.mask * raw_w
            setattr(self.embed, 'weight', w)
        else:
            setattr(self.embed, 'weight', raw_w)

    def forward(self, words):
        self.set_weight()
        return self.embed(words)




if __name__ == '__main__':
    V = 50
    h = 4
    bptt = 10
    batch_size = 2

    embed = torch.nn.Embedding(V, h)

    words = np.random.random_integers(low=0, high=V-1, size=(batch_size, bptt))
    words = torch.LongTensor(words)

    origX = embed(words)
    X = embedded_dropout(embed, words)

    print(origX)
    print(X)
