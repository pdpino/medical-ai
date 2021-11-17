"""Show, attend and tell reproduction.

- Modified from: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
- Modified from: https://github.com/AaronCCWong/Show-Attend-and-Tell
- Very similar to lstm-att-v2, with few nuances
    - AttentionTwoLayers is the same, works fine
"""

from itertools import count
import logging
import torch
from torch import nn
import torch.nn.functional as F


from medai.models.report_generation.att_2layer import AttentionTwoLayers
from medai.utils.nlp import START_IDX, END_IDX
from medai.models.report_generation.word_embedding import create_word_embedding

LOGGER = logging.getLogger(__name__)


class OutWordFC(nn.Module):
    """Equation 7 of the paper (arxiv version)."""
    def __init__(self, embedding_size, hidden_size, features_size, vocab_size):
        super().__init__()

        self.L_h = nn.Linear(hidden_size, embedding_size)
        self.L_z = nn.Linear(features_size, embedding_size)
        self.L_o = nn.Linear(embedding_size, vocab_size)

        self.tanh = nn.Tanh()

    def forward(self, input_words, h_state_t, context_vector):
        # shapes:
        # input_words - bs, embedding_size
        # h_state_t - bs, hidden_size
        # context_vector - bs, features_size

        out2 = self.L_h(h_state_t)
        out3 = self.L_z(context_vector)
        out_words = self.L_o(self.tanh(input_words + out2 + out3))
        # shape: bs, vocab_size
        return out_words


class ShowAttendTellDecoder(nn.Module):
    implemented_dropout = True # dropout_recursive is not implemented though!!!

    def __init__(self, vocab, embedding_size, hidden_size,
                 features_size, dropout_out=0,
                 double_bias=False, attention_size=100,
                 embedding_kwargs={},
                 teacher_forcing=True, **unused_kwargs):
        super().__init__()

        self.vocab_size = len(vocab)
        self.hidden_size = hidden_size
        self.teacher_forcing = teacher_forcing
        self.start_idx = torch.tensor(START_IDX) # pylint: disable=not-callable

        self.features_reduction = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.features_fc = nn.Linear(features_size, hidden_size * 2)
        ## features_fc replaces init_h and init_c
        # self.init_h = nn.Linear(features_size, hidden_size)
        # self.init_c = nn.Linear(features_size, hidden_size)
        self.tanh = nn.Tanh()

        self.f_beta = nn.Linear(hidden_size, 1)

        self.word_embeddings, self.word_embeddings_bn = create_word_embedding(
            vocab,
            embedding_size,
            **embedding_kwargs,
        )
        self.word_lstm = nn.LSTMCell(embedding_size + features_size, hidden_size)
        self.word_fc = OutWordFC(
            embedding_size, hidden_size, features_size, self.vocab_size,
        )

        self.attention = AttentionTwoLayers(
            features_size, hidden_size, internal_size=attention_size,
            double_bias=double_bias,
        )

        self.dropout_out = nn.Dropout(p=dropout_out)

    def get_init_lstm_state(self, image_features):
        # image_features shape: batch_size, features_size, height, width

        initial_state = self.features_fc(self.features_reduction(image_features))
        # shape: batch_size, hidden_size*2
        initial_state = self.tanh(initial_state)
        # shape: batch_size, hidden_size*2

        initial_h_state = initial_state[:, :self.hidden_size]
        initial_c_state = initial_state[:, self.hidden_size:]

        return initial_h_state, initial_c_state


    def forward(self, image_features, reports=None, free=False, max_words=10000, beam_size=0):
        if beam_size is not None and beam_size > 0:
            assert reports is None and free
            return self.caption(image_features, beam_size=beam_size, max_words=max_words)

        batch_size = image_features.size(0)
        # image_features shape: batch_size, features_size, height, width

        device = image_features.device

        # Build initial state
        h_state_t, c_state_t = self.get_init_lstm_state(image_features)

        # Decide teacher forcing
        teacher_forcing = self.teacher_forcing \
            and self.training \
            and reports is not None \
            and not free

        # Set iteration maximum
        if free:
            words_iterator = range(max_words) if max_words else count()
            # pylint: disable=not-callable
            should_stop = torch.tensor(False, device=device).repeat(batch_size)
        else:
            assert reports is not None, 'Cannot pass free=False and reports=None'
            actual_max_len = reports.size(-1)
            words_iterator = range(actual_max_len)
            should_stop = None

        # Build initial inputs
        next_words_indices = self.start_idx.to(device).repeat(batch_size) # shape: batch_size

        # Generate word by word
        seq_out = []
        scores_out = []

        for word_i in words_iterator:
            # Pass state through attention
            att_features, att_scores = self.attention(image_features, h_state_t)
                # att_features shape: batch_size, features_size
                # att_scores shape: batch_size, height, width
            scores_out.append(att_scores)

            # Gating and beta
            beta = torch.sigmoid(self.f_beta(h_state_t)) # shape: batch_size, 1
            gated_att_features = beta * att_features # shape: batch_size, features_size

            # Embed words
            input_words = self.word_embeddings(next_words_indices)
            input_words = self.word_embeddings_bn(input_words)
            # shape: batch_size, embedding_size

            # Concat words and attended image-features
            input_t = torch.cat((input_words, gated_att_features), dim=1)
                # shape: batch_size, embedding_size + features_size


            # Pass thru LSTM
            h_state_t, c_state_t = self.word_lstm(input_t, (h_state_t, c_state_t))
            # shapes: batch_size, hidden_size

            # Predict with FC
            prediction_t = self.word_fc(
                input_words, # shape: batch_size, embedding_size
                self.dropout_out(h_state_t), # shape: batch_size, hidden_size,
                gated_att_features, # shape: batch_size, features_size
            )
            # prediction_t shape: batch_size, vocab_size
            seq_out.append(prediction_t)

            # Decide if should stop
            # Remember if each element in the batch has outputted the END token
            if free:
                is_end_prediction = prediction_t.argmax(dim=-1) == END_IDX # shape: batch_size
                should_stop |= is_end_prediction

                if should_stop.all():
                    break

            # Get input for next word
            if teacher_forcing:
                next_words_indices = reports[:, word_i]
                # shape: batch_size
            else:
                _, next_words_indices = prediction_t.max(dim=1)
                # shape: batch_size


        seq_out = torch.stack(seq_out, dim=1)
        # shape: batch_size, max_sentence_len, vocab_size

        scores_out = torch.stack(scores_out, dim=1)
        # shape: batch_size, max_sentence_len, height, width

        return seq_out, scores_out


    def caption(self, image_features, beam_size=3, max_words=100, debug=False):
        batch_size, features_size, height, width = image_features.size()
        assert batch_size == 1 # FOR NOW

        device = image_features.device

        # We'll treat the problem as having a batch size of k
        encoder_out = image_features.expand(beam_size, features_size, height, width)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.zeros(beam_size, 1,
            device=device, dtype=torch.long,
        ).fill_(START_IDX)
        # shape: (beam_size, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (beam_size, 1)

        # Tensor to store top k sequences' scores; now they're just 0 (dummy init)
        top_k_scores = torch.zeros(beam_size, 1, device=device)

        # Tensor to store top k sequences' alphas; now they're just 1s (dummy init)
        seqs_alpha = torch.ones(beam_size, 1, height, width, device=device)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()
        complete_seqs_alpha = list()

        # Start decoding
        h_state_t, c_state_t = self.get_init_lstm_state(encoder_out)

        # k_remaining_seqs: number less than or equal to beam_size,
        # because sequences are removed from this process once they hit <end>
        k_remaining_seqs = beam_size
        for step in range(max_words):
            embeddings = self.word_embeddings(k_prev_words).squeeze(1)
            # (k_remaining_seqs, embedding_size)

            att_features, att_scores = self.attention(encoder_out, h_state_t)
            # shapes: (k, hidden_size), (k, height, width)

            beta = torch.sigmoid(self.f_beta(h_state_t)) # shape: k, 1
            gated_att_features = beta * att_features # shape: k, features_size

            h_state_t, c_state_t = self.word_lstm(
                torch.cat([embeddings, gated_att_features], dim=1),
                (h_state_t, c_state_t),
            )
            # (k_remaining_seqs, hidden_size)

            # Predict with FC
            scores = self.word_fc(
                embeddings, # shape: k_remaining_seqs, embedding_size
                h_state_t, # shape: k_remaining_seqs, hidden_size,
                gated_att_features, # shape: k_remaining_seqs, features_size
            )
            scores = F.log_softmax(scores, dim=1)
            # shape: k_remaining_seqs, vocab_size

            # Add with scores from the sequence
            scores = top_k_scores.expand_as(scores) + scores  # (k, vocab_size)

            # For the first step
            # all k points will have the same scores (since same k previous words, h, c)
            if step == 0:
                top_k_scores, top_k_words = scores[0].topk(k_remaining_seqs, 0, True, True)
                # shape: k_remaining_seqs
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k_remaining_seqs, 0, True, True)
                # shape: k_remaining_seqs

            # Convert unrolled indices to actual indices of scores
            prev_seq_inds = top_k_words.floor_divide(self.vocab_size).type(torch.long)
            next_word_inds = (top_k_words % self.vocab_size).type(torch.long)
            # shapes: k_remaining_seqs

            # Add new words to sequences
            seqs = torch.cat([
                seqs[prev_seq_inds], # k_remaining_seqs, n_steps
                next_word_inds.unsqueeze(1), # k_remaining_seqs, 1
            ], dim=1)
            # (k_remaining_seqs, step+1)

            # Add alphas to sequences
            seqs_alpha = torch.cat([
                seqs_alpha[prev_seq_inds], # k_remaining_seqs, n_steps, height, width
                att_scores[prev_seq_inds].unsqueeze(1), # k_remaining_seqs, 1, height, width
            ], dim=1)
            # (k_remaining_sesq, step+1, height, width)


            # Which sequences are incomplete (did not reach <end>)?
            incomplete_inds = [
                ind
                for ind, next_word in enumerate(next_word_inds)
                if next_word != END_IDX
            ]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds])
                complete_seqs_scores.extend(top_k_scores[complete_inds])
                complete_seqs_alpha.extend(seqs_alpha[complete_inds])
            k_remaining_seqs -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k_remaining_seqs <= 0:
                if k_remaining_seqs < 0:
                    LOGGER.warning('Off by n error: %d', k_remaining_seqs)
                break

            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            h_state_t = h_state_t[prev_seq_inds[incomplete_inds]]
            c_state_t = c_state_t[prev_seq_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_seq_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        if len(complete_seqs) == 0:
            # Edge case: no report actually finished with <END>
            complete_seqs.extend(seqs)
            complete_seqs_scores.extend(top_k_scores)
            complete_seqs_alpha.extend(seqs_alpha)

            if len(complete_seqs) == 0:
                # Edge case: still no reports finished (i.e. no reports generated at all)
                return [], [], []

        if debug:
            return complete_seqs, complete_seqs_scores, complete_seqs_alpha

        complete_seqs_scores = torch.stack(complete_seqs_scores)
        assert complete_seqs_scores.ndim == 1, f'scores wrong size={complete_seqs_scores.size()}'

        i = complete_seqs_scores.argmax()
        seq = complete_seqs[i] # shape: n_words_generated
        alphas = complete_seqs_alpha[i] # shape: n_words_generated, height, width

        return seq, alphas
