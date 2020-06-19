import torch
from torch import nn

from mrg.utils import PAD_IDX, START_IDX


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, teacher_forcing=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.start_idx = torch.tensor(START_IDX)

        self.embeddings_table = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_IDX)
        self.lstm_cell = torch.nn.LSTMCell(embedding_size, hidden_size)
        self.W_vocab = torch.nn.Linear(hidden_size, vocab_size)
        
        self.teacher_forcing = teacher_forcing

    def forward(self, initial_state, max_sentence_length, reports=None):
        batch_size, hidden_size = initial_state.size()

        # Build initial state
        device = initial_state.device
        initial_c_state = torch.zeros(batch_size, self.hidden_size).to(device)
        state = (initial_state, initial_c_state)

        # Build initial input
        start_idx = self.start_idx.to(device).repeat(batch_size) # shape: batch_size
        input_t = self.embeddings_table(start_idx)
        # shape: batch_size, embedding_size

        # Decide teacher forcing
        teacher_forcing = self.teacher_forcing and self.training and reports is not None
        
        # Generate word by word
        seq_out = []

        for word_i in range(max_sentence_length):
            # Pass thru LSTM
            state = self.lstm_cell(input_t, state)
            h_t, c_t = state
            # shapes: batch_size, hidden_size

            # Predict with FC
            prediction_t = self.W_vocab(h_t) # shape: batch_size, vocab_size
            seq_out.append(prediction_t)

            # Get input for next word
            if teacher_forcing:
                next_words_indices = reports[:, word_i]
                # shape: batch_size
            else:
                _, next_words_indices = prediction_t.max(dim=1)
                # shape: batch_size

            input_t = self.embeddings_table(next_words_indices)
            # shape: batch_size, embedding_size

        return torch.stack(seq_out, dim=1),