"""Difference with decoder_lstm_att.py (i.e. v1):
* Attented features are passed as input to the LSTM, instead of merging with h_state
"""

from itertools import count
import torch
from torch import nn

from medai.models.report_generation.att_2layer import AttentionTwoLayers
from medai.utils.nlp import PAD_IDX, START_IDX, END_IDX


class LSTMAttDecoderV2(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size,
                 features_size,
                 teacher_forcing=True, **unused):
        super().__init__()

        self.hidden_size = hidden_size
        self.teacher_forcing = teacher_forcing
        self.start_idx = torch.tensor(START_IDX)

        self.embeddings_table = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_IDX)
        self.lstm_cell = nn.LSTMCell(embedding_size + features_size, hidden_size)
        self.W_vocab = nn.Linear(hidden_size, vocab_size)

        self.attention = AttentionTwoLayers(features_size, hidden_size, double_bias=False)


    def forward(self, image_features, reports=None, free=False, max_words=10000):
        batch_size = image_features.size()[0]
            # image_features shape: batch_size, features_size, height, width

        device = image_features.device

        # Build initial state
        initial_c_state = torch.zeros(batch_size, self.hidden_size).to(device)
        initial_h_state = torch.zeros(batch_size, self.hidden_size).to(device)
        att_features, att_scores = self.attention(image_features, initial_h_state)
            # att_features shape: batch_size, features_size
            # att_scores shape: batch_size, height, width

        state = (initial_h_state, initial_c_state)

        # Build initial input
        start_idx = self.start_idx.to(device).repeat(batch_size) # shape: batch_size
        input_words = self.embeddings_table(start_idx)
            # shape: batch_size, embedding_size
        input_t = torch.cat((input_words, att_features), dim=1)
            # shape: batch_size, embedding_size + features_size

        # Decide teacher forcing
        teacher_forcing = self.teacher_forcing \
            and self.training \
            and reports is not None \
            and not free
        
        # Set iteration maximum
        if free:
            words_iterator = range(max_words) if max_words else count()
            should_stop = torch.tensor(False).to(device).repeat(batch_size)
        else:
            assert reports is not None, 'Cant pass free=False and reports=None'
            actual_max_len = reports.size()[-1]
            words_iterator = range(actual_max_len)
            should_stop = None

        # Generate word by word
        seq_out = []
        scores_out = []

        for word_i in words_iterator:
            # Pass thru LSTM
            state = self.lstm_cell(input_t, state)
            h_t, c_t = state
            # shapes: batch_size, hidden_size

            # Predict with FC
            prediction_t = self.W_vocab(h_t) # shape: batch_size, vocab_size
            seq_out.append(prediction_t)

            # Pass state thru attention
            att_features, att_scores = self.attention(image_features, h_t)
            scores_out.append(att_scores)

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

            input_words = self.embeddings_table(next_words_indices)
                # shape: batch_size, embedding_size
            input_t = torch.cat((input_words, att_features), dim=1)
                # shape: batch_size, embedding_size + features_size

        seq_out = torch.stack(seq_out, dim=1)
        # shape: batch_size, max_sentence_len, vocab_size

        scores_out = torch.stack(scores_out, dim=1)
        # shape: batch_size, max_sentence_len, height, width

        return seq_out, scores_out