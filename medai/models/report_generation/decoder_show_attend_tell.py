"""Show, attend and tell reproduction.

- Modified from: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
- Modified from: https://github.com/AaronCCWong/Show-Attend-and-Tell
- Very similar to lstm-att-v2, with few nuances
    - AttentionTwoLayers is the same, works fine
"""

from itertools import count
import torch
from torch import nn

from medai.models.report_generation.att_2layer import AttentionTwoLayers
from medai.utils.nlp import START_IDX, END_IDX
from medai.models.report_generation.word_embedding import create_word_embedding

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
            embedding_size, hidden_size, features_size, len(vocab),
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


    def forward(self, image_features, reports=None, free=False, max_words=10000):
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
