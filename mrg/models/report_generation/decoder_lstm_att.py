import torch
from torch import nn

from mrg.utils.nlp import PAD_IDX, START_IDX
from mrg.models.report_generation.att_2layer import AttentionTwoLayers


class LSTMAttDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size,
                 features_size,
                 teacher_forcing=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.teacher_forcing = teacher_forcing
        self.start_idx = torch.tensor(START_IDX)

        self.embeddings_table = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_IDX)
        self.lstm_cell = nn.LSTMCell(embedding_size, hidden_size)
        self.W_vocab = nn.Linear(hidden_size, vocab_size)

        self.attention = AttentionTwoLayers(features_size, hidden_size)
        self.att_to_state = nn.Linear(features_size[0], hidden_size)


    def forward(self, image_features, max_sentence_length, reports=None):
        batch_size = image_features.size()[0]
            # image_features shape: batch_size, n_features, height, width

        device = image_features.device

        # Build initial state
        initial_h_state = torch.zeros(batch_size, self.hidden_size).to(device)
        att_features, att_scores = self.attention(image_features, initial_h_state)
            # att_features shape: batch_size, n_features
            # att_scores features: batch_size, height, width
        initial_c_state = self.att_to_state(att_features)
            # shape: batch_size, hidden_size

        state = (initial_h_state, initial_c_state)

        # Build initial input
        start_idx = self.start_idx.to(device).repeat(batch_size) # shape: batch_size
        input_t = self.embeddings_table(start_idx)
            # shape: batch_size, embedding_size

        # Decide teacher forcing
        teacher_forcing = self.teacher_forcing and self.training and reports is not None
        
        # Generate word by word
        seq_out = []
        scores_out = []

        for word_i in range(max_sentence_length):
            # Pass thru LSTM
            state = self.lstm_cell(input_t, state)
            h_t, c_t = state
            # shapes: batch_size, hidden_size

            # Predict with FC
            prediction_t = self.W_vocab(h_t) # shape: batch_size, vocab_size
            seq_out.append(prediction_t)

            # Pass state thru attention
            att_features, att_scores = self.attention(image_features, h_t)
            c_state = self.att_to_state(att_features)
            state = (h_t, c_state)
            scores_out.append(att_scores)

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

        scores_out = torch.stack(scores_out, dim=1)
        # shape: batch_size, max_sentence_len, height, width

        return seq_out, scores_out