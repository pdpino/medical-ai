"""Difference with decoder_lstm_att.py (i.e. v1):
* Attented features are passed as input to the LSTM, instead of merging with h_state
* h0 and c0 are initialized with features information (same as lstm-v2)
"""

from itertools import count
import torch
from torch import nn
import torch.nn.functional as F

from medai.models.report_generation.att_2layer import AttentionTwoLayers
from medai.utils.nlp import PAD_IDX, START_IDX, END_IDX


class LSTMAttDecoderV2(nn.Module):
    implemented_dropout = True

    def __init__(self, vocab_size, embedding_size, hidden_size,
                 features_size, dropout_out=0, dropout_recursive=0,
                 double_bias=False,
                 teacher_forcing=True, **unused_kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.teacher_forcing = teacher_forcing
        self.start_idx = torch.tensor(START_IDX) # pylint: disable=not-callable

        self.features_reduction = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.features_fc = nn.Linear(features_size, hidden_size * 2)

        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_IDX)
        self.word_lstm = nn.LSTMCell(embedding_size + features_size, hidden_size)
        self.word_fc = nn.Linear(hidden_size, vocab_size)

        self.attention = AttentionTwoLayers(
            features_size, hidden_size, double_bias=double_bias,
        )

        self.dropout_out = dropout_out
        self.dropout_recursive = dropout_recursive


    def forward(self, image_features, reports=None, free=False, max_words=10000):
        batch_size = image_features.size()[0]
            # image_features shape: batch_size, features_size, height, width

        device = image_features.device

        # Build initial state
        initial_state = self.features_fc(self.features_reduction(image_features))
            # shape: batch_size, hidden_size*2
        initial_h_state = initial_state[:, :self.hidden_size]
        initial_c_state = initial_state[:, self.hidden_size:]
            # shapes: batch_size, hidden_size

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
            assert reports is not None, 'Cant pass free=False and reports=None'
            actual_max_len = reports.size()[-1]
            words_iterator = range(actual_max_len)
            should_stop = None

        # Build initial inputs
        next_words_indices = self.start_idx.to(device).repeat(batch_size) # shape: batch_size
        h_state_t = initial_h_state
        c_state_t = initial_c_state

        # Generate word by word
        seq_out = []
        scores_out = []

        for word_i in words_iterator:
            # Pass state through attention
            att_features, att_scores = self.attention(image_features, h_state_t)
                # att_features shape: batch_size, features_size
                # att_scores shape: batch_size, height, width
            scores_out.append(att_scores)

            # Embed words
            input_words = self.word_embeddings(next_words_indices)
                # shape: batch_size, embedding_size

            # Concat words and attended image-features
            input_t = torch.cat((input_words, att_features), dim=1)
                # shape: batch_size, embedding_size + features_size


            # Pass thru LSTM
            h_state_t, c_state_t = self.word_lstm(input_t, (h_state_t, c_state_t))
            # shapes: batch_size, hidden_size

            # Pass thru out-dropout, if any
            out_h_t = h_state_t
            if self.dropout_out:
                out_h_t = F.dropout(out_h_t, self.dropout_out, training=self.training)

            # Predict with FC
            prediction_t = self.word_fc(out_h_t) # shape: batch_size, vocab_size
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

            # Apply recursive dropout
            if self.dropout_recursive:
                h_state_t = F.dropout(h_state_t, self.dropout_recursive, training=self.training)
                c_state_t = F.dropout(c_state_t, self.dropout_recursive, training=self.training)


        seq_out = torch.stack(seq_out, dim=1)
        # shape: batch_size, max_sentence_len, vocab_size

        scores_out = torch.stack(scores_out, dim=1)
        # shape: batch_size, max_sentence_len, height, width

        return seq_out, scores_out
