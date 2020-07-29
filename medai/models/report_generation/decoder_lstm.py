from itertools import count
import torch
from torch import nn
import numpy as np

from medai.utils.nlp import PAD_IDX, START_IDX, END_IDX


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size,
                 features_size,
                 teacher_forcing=True, **kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.teacher_forcing = teacher_forcing
        self.start_idx = torch.tensor(START_IDX)

        self.features_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(features_size, hidden_size),
        )

        self.embeddings_table = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_IDX)
        self.lstm_cell = nn.LSTMCell(embedding_size, hidden_size)
        self.W_vocab = nn.Linear(hidden_size, vocab_size)

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
        initial_h_state = self.features_fc(features)
            # shape: batch_size, hidden_size

        # Build initial state
        initial_c_state = torch.zeros(batch_size, self.hidden_size).to(device)
        state = (initial_h_state, initial_c_state)

        # Build initial input
        start_idx = self.start_idx.to(device).repeat(batch_size) # shape: batch_size
        input_t = self.embeddings_table(start_idx)
            # shape: batch_size, embedding_size

        # Decide teacher forcing
        teacher_forcing = self.teacher_forcing \
            and self.training \
            and reports is not None \
            and not free \

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

        for word_i in words_iterator:
            # Pass thru LSTM
            state = self.lstm_cell(input_t, state)
            h_t, c_t = state
            # shapes: batch_size, hidden_size

            # Predict with FC
            prediction_t = self.W_vocab(h_t) # shape: batch_size, vocab_size
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

            input_t = self.embeddings_table(next_words_indices)
            # shape: batch_size, embedding_size

        seq_out = torch.stack(seq_out, dim=1)
        # shape: batch_size, max_sentence_len, vocab_size

        return seq_out,