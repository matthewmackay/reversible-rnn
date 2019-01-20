import pdb

import torch
import torch.nn as nn

from onmt.modules.UtilClass import BottleLinear
from onmt.Utils import aeq, sequence_mask


class MultiSizeAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `hidden_size`
    and a source matrix `H` of size `n x hidden_size`, to an output
    of size `hidden_size`.

    All models compute the output as
    :math:`c = \sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       hidden_size (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """
    def __init__(self, hidden_size, context_size, attn_type="dot"):
        super(MultiSizeAttention, self).__init__()

        self.hidden_size = hidden_size
        self.context_size = context_size

        self.attn_type = attn_type
        assert (self.attn_type in ['dot', 'general', 'mlp', 'mlp-conc']), (
                "Please select a valid attention type.")

        if self.attn_type == 'mlp-conc':
            # Maps hidden_size + context_size --> 1
            self.mlp_conc = nn.Linear(hidden_size + context_size, 1, bias=False)
        elif self.attn_type == 'general':
            # self.linear_in = nn.Linear(hidden_size, hidden_size, bias=False)
            self.linear_in = nn.Linear(hidden_size, context_size, bias=False)
        elif self.attn_type == 'mlp':
            self.linear_context = BottleLinear(hidden_size, hidden_size, bias=False)
            self.linear_query = nn.Linear(hidden_size, hidden_size, bias=True)
            self.v = BottleLinear(hidden_size, 1, bias=False)

        # mlp wants it with bias
        out_bias = self.attn_type == 'mlp'
        # self.linear_out = nn.Linear(hidden_size*2, hidden_size, bias=out_bias)
        self.linear_out = nn.Linear(hidden_size + context_size, hidden_size, bias=out_bias)

        self.sm = nn.Softmax()
        self.tanh = nn.Tanh()

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x hidden_size]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x context_size]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        # aeq(src_dim, tgt_dim)
        aeq(self.context_size, src_dim)
        aeq(self.hidden_size, tgt_dim)

        if self.attn_type == 'mlp-conc':

            attn_result_list = []

            for i in range(tgt_batch):
                query_mat = h_t[i]   # tgt_len x tgt_dim   14 x 300
                source_mat = h_s[i]  # src_len x src_dim   9 x 300

                # repeated_source_mat = source_mat.repeat(rep2, 1)
                # repeated_query_mat = torch.cat(query_mat.unsqueeze(1).expand(query_mat.size(0), src_len, query_mat.size(1)))
                repeated_query_mat = query_mat.unsqueeze(1).expand(query_mat.size(0), src_len, query_mat.size(1)).contiguous().view(-1,tgt_dim)
                repeated_source_mat = source_mat.repeat(tgt_len, 1)
                combined = torch.cat([repeated_query_mat, repeated_source_mat], dim=1)
                res = self.mlp_conc(combined)
                res = res.view(tgt_len, src_len)

                attn_result_list.append(res)

            return torch.stack(attn_result_list)

        elif self.attn_type == 'context_f':
            # Apply a linear transformation on the context vectors and dot product them
            pass

        elif self.attn_type in ['general', 'dot']:
            if self.attn_type == 'general':
                h_t_ = h_t.view(tgt_batch*tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                # h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
                h_t = h_t_.view(tgt_batch, tgt_len, src_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            hidden_size = self.hidden_size
            wq = self.linear_query(h_t.view(-1, hidden_size))
            wq = wq.view(tgt_batch, tgt_len, 1, hidden_size)
            wq = wq.expand(tgt_batch, tgt_len, src_len, hidden_size)

            uh = self.linear_context(h_s.contiguous().view(-1, hidden_size))
            uh = uh.view(src_batch, 1, src_len, hidden_size)
            uh = uh.expand(src_batch, tgt_len, src_len, hidden_size)

            # (batch, t_len, s_len, d)
            wquh = self.tanh(wq + uh)

            return self.v(wquh.view(-1, hidden_size)).view(tgt_batch, tgt_len, src_len)

    def forward(self, input, context, context_lengths=None, coverage=None):
        """

        Args:
          input (`FloatTensor`): query vectors `[batch x tgt_len x hidden_size]`
          context (`FloatTensor`): source vectors `[batch x src_len x hidden_size]`
          context_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x hidden_size]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """

        batch, sourceL, context_size = context.size()
        batch_, targetL, hidden_size = input.size()
        aeq(batch, batch_)

        # compute attention scores, as in Luong et al.
        align = self.score(input, context)  # BS x tgt_len x src_len   64 x 19 x 13

        # pdb.set_trace()

        if context_lengths is not None:
            mask = sequence_mask(context_lengths)
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.data.masked_fill_(1 - mask, -float('inf'))

        # Softmax to normalize attention weights
        align_vectors = self.sm(align.view(batch*targetL, sourceL))
        align_vectors = align_vectors.view(batch, targetL, sourceL)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, context)

        # concatenate
        concat_c = torch.cat([c, input], 2).view(batch*targetL, -1)
        attn_h = self.linear_out(concat_c).view(batch, targetL, hidden_size)
        if self.attn_type in ["general", "dot"]:
            attn_h = self.tanh(attn_h)

        attn_h = attn_h.transpose(0, 1).contiguous()
        align_vectors = align_vectors.transpose(0, 1).contiguous()

        # Check output sizes
        targetL_, batch_, dim_ = attn_h.size()
        # aeq(targetL, targetL_)
        # aeq(batch, batch_)
        # aeq(hidden_size, dim_)
        targetL_, batch_, sourceL_ = align_vectors.size()
        # aeq(targetL, targetL_)
        # aeq(batch, batch_)
        # aeq(sourceL, sourceL_)

        return attn_h, align_vectors
