import ipdb

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import onmt.translate.Beam
import onmt.io


class Translator(object):
    """
    Uses a model to translate a batch of sentences.

    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
    """
    def __init__(self, model, fields,
                 beam_size, n_best=1,
                 max_length=100,
                 global_scorer=None, copy_attn=False, cuda=False,
                 beam_trace=False, min_length=0):
        self.model = model
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.copy_attn = copy_attn
        self.beam_size = beam_size
        self.cuda = cuda
        self.min_length = min_length

        # for debugging
        self.beam_accum = None
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def translate_batch(self, batch, data):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        beam_size = self.beam_size
        batch_size = batch.batch_size
        data_type = data.data_type
        vocab = self.fields["tgt"].vocab
        beam = [onmt.translate.Beam(beam_size, n_best=self.n_best,
                                    cuda=self.cuda,
                                    global_scorer=self.global_scorer,
                                    pad=vocab.stoi[onmt.io.PAD_WORD],
                                    eos=vocab.stoi[onmt.io.EOS_WORD],
                                    bos=vocab.stoi[onmt.io.BOS_WORD],
                                    min_length=self.min_length)
                for __ in range(batch_size)]

        # (1) Run the encoder on the src.
        src = onmt.io.make_features(batch, 'src', data_type)
        src_lengths = None
        if data_type == 'text':
            _, src_lengths = batch.src

        # src: 29 x 30 x 1
        # src_lengths: 30

        if self.model.opt.encoder_model == 'Vanilla' and self.model.opt.decoder_model == 'Vanilla':  # Vanilla LSTM or GRU

            src = src.squeeze()  # 29 x 30

            enc_states, context = self.model.encoder(src, src_lengths)  # FOR VANILLA MODEL
            #   enc_states: Variable 1 x 30 x 300
            #   context: Variable 29 x 39 x 300
            #   enc_output_dict: dict with keys 'actual_usage', 'naive_usage', 'optimal_usage'

            dec_states = self.model.decoder.init_decoder_state(enc_states)
            #   dec_states: RNNDecoderState
            #   type(dec_states._all) == <class 'list'>
            #   len(dec_states._all) == 1
            #   type(dec_states.hidden) == <class 'list'>
            #   len(dec_states.hidden) == 1

            # (2) Repeat src objects `beam_size` times.
            context = Variable(context.data.repeat(1, beam_size, 1), volatile=True)
            #   context.shape (BEFORE REPEATING) == 29 x 30 x 320
            #   context.shape (AFTER REPEATING) == 29 x 150 x 320   (150 comes from 30 * beam_size, where beam_size is 5)

            context_lengths = src_lengths.repeat(beam_size)
            #   src_lengths.shape == 30
            #   context_lengths (with the repeat) has shape 150  (which is 30 * beam_size, where beam_size is 5)

            vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True) for e in dec_states._all]
            dec_states.hidden = tuple(vars)  # After the repeating, dec_states.hidden[0] has shape 1 x 150 x 600


            # (3) run the decoder to generate sentences, using beam search.
            for i in range(self.max_length):
                if all((b.done() for b in beam)):  # IMPORTANT: beam is a list containing 30 Beam objects, which is equal to the batch size!
                    break

                # Construct batch x beam_size nxt words.
                # Get all the pending current beam words and arrange for forward.
                inp = Variable(torch.stack([b.get_current_state() for b in beam]).t().contiguous().view(1, -1), volatile=True)
                #    b.get_current_state() returns a LongTensor of size 5 (which is the beam_size) --> these are "outputs for the current timestep"
                #   inp: LongTensor with shape 1 x 150

                # Run one step.
                dec_out, dec_states, attn = self.model.decoder(inp, context, dec_states, context_lengths=context_lengths)  # FOR VANILLA MODEL
                #   inp.shape == 1 x 150
                #   context.shape == 29 x 150 x 300
                #   dec_states: DecoderState
                #   dec_out: FloatTensor with shape 1 x 150 x 300
                #   dec_states.hidden: tuple containing 1 element, a FloatTensor of shape 1 x 150 x 300
                #   attn: dict with one key "std" that has value FloatTensor 1 x 150 x 29

                dec_out = dec_out.squeeze(0)  # AFTER SQUEEZE: 150 x 300
                out = F.log_softmax(self.model.decoder.generator.forward(dec_out).data)  # out.shape == 150 x 18563
                out = out.view(beam_size, batch_size, -1)  # Equivalent to unbottle(out). Final shape of out is 5 x 30 x 18563

                # (c) Advance each beam.
                for j, b in enumerate(beam):
                    b.advance(out[:, j], attn["std"].view(beam_size, batch_size, -1).data[:, j, :context_lengths[j]])  # <-- check why attention is passed in here

                    idx = j
                    positions = b.get_current_origin()

                    for e in dec_states._all:
                        a, br, d = e.size()  # 1 x 150 x 300
                        sent_states = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
                        #   e.view(a, beam_size, br // beam_size, d) has shape 1 x 5 x 30 x 300
                        #   br // beam_size == 30
                        #   sent_states has shape 1 x 5 x 300 (where 5 is the beam_size)

                        sent_states.data.copy_(sent_states.data.index_select(1, positions))
                        #   sent_states still has shape 1 x 5 x 300 in the end

            # (4) Extract sentences from beam.
            ret = self._from_beam(beam)
            ret["gold_score"] = [0] * batch_size
            if "tgt" in batch.__dict__:
                ret["gold_score"] = self._run_target(batch, data)
            ret["batch"] = batch
            return ret

        elif self.model.opt.encoder_model == 'Rev' and self.model.opt.decoder_model == 'Rev':  # RevLSTM or RevGRU

            batch_size = src.size(1)  # 30
            enc_input_seq = src.squeeze()  # 29 x 30
            enc_lengths = src_lengths

            hiddens = self.model.encoder.init_hiddens(batch_size)  # hiddens is a list; hiddens[0] has shape 30 x 600 for the RevLSTM
            enc_states, context = self.model.encoder.forward_test(enc_input_seq, enc_lengths, hiddens)
            #   enc_states: list with 1 elem; enc_states[0] has shape 30 x 600 for RevLSTM
            #   context: 29 x 30 x 320
            #   enc_buffers: (None, None, None, None)
            #   main_buf: InformationBuffer
            #   slice_buf: InformationBuffer
            #   enc_dict: dictionary with keys 'optimal_bits' and 'normal_bits'

            # (2) Repeat src objects `beam_size` times.
            context = Variable(context.data.repeat(1, beam_size, 1), volatile=True)
            #   context.shape (BEFORE REPEATING) == 29 x 30 x 320
            #   context.shape (AFTER REPEATING) == 29 x 150 x 320   (150 comes from 30 * beam_size, where beam_size is 5)

            context_lengths = src_lengths.repeat(beam_size)
            #   src_lengths.shape == 30
            #   context_lengths (with the repeat) has shape 150  (which is 30 * beam_size, where beam_size is 5)

            vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True) for e in enc_states]
            dec_states = vars  # After the repeating, dec_states.hidden[0] has shape 1 x 150 x 600

            # (3) run the decoder to generate sentences, using beam search.
            for i in range(self.max_length):
                if all((b.done() for b in beam)):  # IMPORTANT: beam is a list containing 30 Beam objects, which is equal to the batch size!
                    break

                # Construct batch x beam_size nxt words.
                # Get all the pending current beam words and arrange for forward.
                inp = Variable(torch.stack([b.get_current_state() for b in beam]).t().contiguous().view(1, -1), volatile=True)
                #    b.get_current_state() returns a LongTensor of size 5 (which is the beam_size) --> these are "outputs for the current timestep"
                #   inp: LongTensor with shape 1 x 150

                dec_states = [e.squeeze() for e in dec_states]

                # Run one step.
                dec_out, dec_states, attn = self.model.decoder.forward_test(inp, dec_states, context, context_lengths)
                #   inp.shape == 1 x 150
                #   context.shape == 29 x 150 x 300
                #   dec_states: DecoderState
                #   dec_out: FloatTensor with shape 1 x 150 x 300
                #   dec_states.hidden: tuple containing 1 element, a FloatTensor of shape 150 x 600
                #   attn: dict with one key "std" that has value FloatTensor 1 x 150 x 29

                dec_out = dec_out.squeeze(0)  # AFTER SQUEEZE: 150 x 300
                out = F.log_softmax(dec_out.data)  # out.shape == 150 x 18563
                out = out.view(beam_size, batch_size, -1)  # Equivalent to unbottle(out). Final shape of out is 5 x 30 x 18563

                # (c) Advance each beam.
                for j, b in enumerate(beam):
                    b.advance(out[:, j], attn["std"].view(beam_size, batch_size, -1).data[:, j, :context_lengths[j]])  # <-- check why attention is passed in here

                    idx = j
                    positions = b.get_current_origin()

                    for e in dec_states:
                        br, d = e.size()  # 1 x 150 x 300
                        sent_states = e.view(beam_size, br // beam_size, d)[:, idx, :]
                        #   e.view(a, beam_size, br // beam_size, d) has shape 5 x 30 x 600
                        #   br // beam_size == 30
                        #   sent_states has shape 1 x 5 x 300 (where 5 is the beam_size)
                        sent_states.data.copy_(sent_states.data.index_select(0, positions))
                        #   sent_states still has shape 1 x 5 x 300 in the end

            # (4) Extract sentences from beam.
            ret = self._from_beam(beam)
            ret["gold_score"] = [0] * batch_size
            if "tgt" in batch.__dict__:
                ret["gold_score"] = self._run_target(batch, data)
            ret["batch"] = batch
            return ret

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": [],
               "attention": []}
        for b in beam:
            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
            ret["attention"].append(attn)
        return ret

    def _run_target(self, batch, data):
        data_type = data.data_type
        _, src_lengths = batch.src

        src = onmt.io.make_features(batch, 'src', data_type)  # src.shape == 29 x 30 x 1
        tgt_in = onmt.io.make_features(batch, 'tgt')[:-1]  # tgt_in.shape == 28 x 30 x 1

        if self.model.opt.encoder_model == 'Vanilla' and self.model.opt.decoder_model == 'Vanilla':  # Vanilla LSTM or GRU

            src = src.squeeze()  # 29 x 30
            tgt_in = tgt_in.squeeze()  # 28 x 30

            enc_states, context = self.model.encoder(src, src_lengths)
            #   enc_states.shape == 1 x 30 x 300
            #   context.shape == 29 x 30 x 300
            dec_states = self.model.decoder.init_decoder_state(enc_states)

            #  (2) if a target is specified, compute the 'goldScore' (i.e. log likelihood) of the target under the model
            tt = torch.cuda if self.cuda else torch
            gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
            dec_out, dec_states, attn = self.model.decoder(tgt_in, context, dec_states, context_lengths=src_lengths)
            #   dec_out.shape == 28 x 30 x 300

            tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
            for dec, tgt in zip(dec_out, batch.tgt[1:].data):
                # Log prob of each word.
                out = F.log_softmax(self.model.decoder.generator.forward(dec))  # dec shape: 30 x 300, out.shape == 30 x 18563
                tgt = tgt.unsqueeze(1)  # after unsqueeze, 30 x 1
                scores = out.data.gather(1, tgt)  # scores.shape == 30 x 1
                scores.masked_fill_(tgt.eq(tgt_pad), 0)  # 30 x 1
                gold_scores += scores.squeeze()  # gold_scores.shape == 30
            return gold_scores
        elif self.model.opt.encoder_model == 'Rev' and self.model.opt.decoder_model == 'Rev':  # RevLSTM or RevGRU

            # src: 29 x 30 x 1
            # tgt_in: 28 x 30 x 1

            batch_size = src.size(1)
            enc_input_seq = src.squeeze()  # 29 x 30
            tgt_in = tgt_in.squeeze()  # 28 x 30
            enc_lengths = src_lengths  # 30

            hiddens = self.model.encoder.init_hiddens(batch_size)  # hiddens is a list; hiddens[0] has shape 30 x 600 for the RevLSTM
            enc_states, context = self.model.encoder.forward_test(enc_input_seq, enc_lengths, hiddens)
            #   enc_states.shape == 30 x 600
            #   context.shape == 29 x 30 x 320

            # dec_states = self.model.decoder.init_decoder_state(enc_states)
            dec_states = enc_states

            #  (2) if a target is specified, compute the 'goldScore' (i.e. log likelihood) of the target under the model
            tt = torch.cuda if self.cuda else torch
            gold_scores = tt.FloatTensor(batch.batch_size).fill_(0)
            #   dec_out.shape == 28 x 30 x 300

            dec_out, dec_states, attn = self.model.decoder.forward_test(tgt_in, dec_states, context, enc_lengths)
            #   dec_out: 840 x 18563
            #   dec_states: 30 x 600
            dec_out = dec_out.view(tgt_in.size(0), tgt_in.size(1), -1)  # 28 x 30 x 18563

            tgt_pad = self.fields["tgt"].vocab.stoi[onmt.io.PAD_WORD]
            for dec, tgt in zip(dec_out, batch.tgt[1:].data):
                # Log prob of each word.
                out = F.log_softmax(dec)  # dec.shape == out.shape == 30 x 18563
                tgt = tgt.unsqueeze(1)  # after unsqueeze, 30x1
                scores = out.data.gather(1, tgt)  # 30x1
                scores.masked_fill_(tgt.eq(tgt_pad), 0)  # 30x1
                gold_scores += scores.squeeze()  # 30
            return gold_scores
