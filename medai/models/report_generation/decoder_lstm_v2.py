"""Difference with v1:
* Initializes both h0 and c0 with the image features.
"""
from itertools import count
import torch
from torch import nn
import torch.nn.functional as F

from medai.utils.nlp import START_IDX, END_IDX
from medai.models.report_generation.word_embedding import create_word_embedding


class LSTMDecoderV2(nn.Module):
    implemented_dropout = True

    def __init__(self, vocab, embedding_size, hidden_size,
                 features_size, dropout_out=0, dropout_recursive=0,
                 embedding_kwargs={},
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

        self.word_embeddings = create_word_embedding(vocab, embedding_size, **embedding_kwargs)
        self.word_lstm = nn.LSTMCell(embedding_size, hidden_size)
        self.word_fc = nn.Linear(hidden_size, len(vocab))

        self.dropout_out = dropout_out
        self.dropout_recursive = dropout_recursive

    def forward(self, features, reports=None, free=False, max_words=10000):
        """Forward pass.

        Args:
            features: tensor of shape (batch_size, n_features, height, width)
            reports: tensor of shape (batch_size, n_words), or None
            free: boolean, indicating if the generation should be free
                (or bounded by the size of reports). If free=False, reports can't be None.
            max_words: int indicating an upper bound to the free-generation. Used to avoid an
                infinite generation loop.
        Returns:
            (seq_out,)
            seq_out: tensor of shape (batch_size, n_generated_words, vocab_size)
                If free=False, n_generated_words == n_words;
                else, n_generated_words may be any number of words.
        """
        batch_size = features.size()[0]
        device = features.device

        # Transform features to correct size
        initial_state = self.features_fc(self.features_reduction(features))
            # shape: batch_size, hidden_size*2
        initial_h_state = initial_state[:, :self.hidden_size]
        initial_c_state = initial_state[:, self.hidden_size:]
            # shapes: batch_size, hidden_size

        # Build initial state
        h_state_t = initial_h_state
        c_state_t = initial_c_state

        # Build initial input
        start_idx = self.start_idx.to(device).repeat(batch_size) # shape: batch_size

        # Decide teacher forcing
        teacher_forcing = self.teacher_forcing \
            and self.training \
            and reports is not None \
            and not free \

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

        # Generate word by word
        seq_out = []
        next_words_indices = start_idx

        for word_i in words_iterator:
            # Pass words thru embedding
            input_t = self.word_embeddings(next_words_indices)
            # shape: batch_size, embedding_size

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

        return (seq_out,)
