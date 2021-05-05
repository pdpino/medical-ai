"""Differences with v1:
* Initialize h0 and c0 with feature information.
"""
from itertools import count
import torch
from torch import nn

from medai.models.report_generation.att_2layer import AttentionTwoLayers
from medai.models.report_generation.att_no_att import NoAttention
from medai.utils.nlp import START_IDX, END_IDX
from medai.models.report_generation.word_embedding import create_word_embedding


class HierarchicalLSTMAttDecoderV2(nn.Module):
    def __init__(self, vocab, embedding_size, hidden_size,
                 features_size, teacher_forcing=True, stop_threshold=0.5,
                 embedding_kwargs={}, return_topics=False,
                 attention=True, double_bias=False, **unused_kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.teacher_forcing = teacher_forcing
        self.start_idx = torch.tensor(START_IDX) # pylint: disable=not-callable
        self.stop_threshold = stop_threshold
        self.return_topics = return_topics # debug and analysis option

        # Attention input
        self._use_attention = attention
        if attention:
            self.attention = AttentionTwoLayers(
                features_size, hidden_size, double_bias=double_bias,
            )
        else:
            self.attention = NoAttention(reduction='mean')

        # To initialize S-LSTM hidden states
        self.features_reduction = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.features_fc = nn.Linear(features_size, hidden_size * 2)

        # Sentence LSTM
        self.sentence_lstm = nn.LSTMCell(features_size, hidden_size)
        self.stop_control = nn.Linear(hidden_size, 1)

        # Word LSTM (REVIEW: reuse code with flat-lstm??)
        self.word_embeddings, self.word_embeddings_bn = create_word_embedding(
            vocab,
            embedding_size,
            **embedding_kwargs,
        )
        self.word_lstm = nn.LSTMCell(embedding_size, hidden_size)
        self.word_fc = nn.Linear(hidden_size, len(vocab))

    def forward(self, features, reports=None, free=False, max_sentences=100, max_words=1000):
        # features shape: batch_size, features_size, height, width
        batch_size = features.size()[0]
        device = features.device

        # Build initial state
        initial_state = self.features_fc(self.features_reduction(features))
            # shape: batch_size, hidden_size*2
        h_state_t = initial_state[:, :self.hidden_size]
        c_state_t = initial_state[:, self.hidden_size:]
            # shapes: batch_size, hidden_size

        # Decide teacher forcing
        teacher_forcing = self.teacher_forcing \
            and self.training \
            and reports is not None \
            and not free

        # Set iteration maximum
        if free:
            sentences_iterator = range(max_sentences) if max_sentences else count()
            # pylint: disable=not-callable
            should_stop = torch.tensor(False, device=device).repeat(batch_size)
        else:
            assert reports is not None, 'Cant pass free=False and reports=None'
            actual_n_sentences = reports.size()[1]
            sentences_iterator = range(actual_n_sentences)
            should_stop = None

        # Iterate over sentences
        seq_out = []
        stops_out = []
        scores_out = []
        topics_out = []

        for sentence_i in sentences_iterator:
            # Get next input
            att_features, att_scores = self.attention(features, h_state_t)
                # att_features shape: batch_size, features_size
                # att_scores features: (batch_size, features-height, features-width), or None
            sentence_input_t = att_features

            if self._use_attention:
                scores_out.append(att_scores)

            # Pass thru LSTM
            h_state_t, c_state_t = self.sentence_lstm(sentence_input_t, (h_state_t, c_state_t))
            # shapes: batch_size, hidden_size

            # Generate topic vector
            topic = h_state_t
            if self.return_topics:
                topics_out.append(topic)

            # Generate sentence with topic
            words = self.generate_sentence(topic,
                                           sentence_i,
                                           reports=reports,
                                           teacher_forcing=teacher_forcing,
                                           free=free,
                                           max_words=max_words,
                                          )
            seq_out.append(words)

            # Decide stop
            stop = torch.sigmoid(self.stop_control(h_state_t)).squeeze(-1) # shape: batch_size
            stops_out.append(stop)
            if free:
                should_stop |= (stop >= self.stop_threshold)

                if should_stop.all():
                    break

        seq_out = torch.stack(seq_out, dim=1)
        # shape: batch_size, n_sentences, max_n_words, vocab_size

        stops_out = torch.stack(stops_out, dim=1)
        # shape: batch_size, n_sentences

        if self._use_attention and len(scores_out) > 0:
            scores_out = torch.stack(scores_out, dim=1)
            # shape: batch_size, n_sentences, height, width
        else:
            scores_out = None

        if self.return_topics:
            topics_out = torch.stack(topics_out, dim=1)
            # shape: batch_size, n_sentences, hidden_size

        # TODO: use a namedtuple??
        return seq_out, stops_out, scores_out, topics_out

    def generate_sentence(self, topic, sentence_i,
                          reports=None, teacher_forcing=True, free=False, max_words=1000):
        # REVIEW: re-use this logic with a WordDecoder ???

        # topic shape: batch_size, hidden_size
        batch_size = topic.size()[0]
        device = topic.device

        # Build initial state
        h_state_t = topic
        c_state_t = torch.zeros(batch_size, self.hidden_size, device=device)

        # Build initial input
        next_words_indices = self.start_idx.to(device).repeat(batch_size)

        words_out = []


        # Set iteration maximum
        if free:
            words_iterator = range(max_words) if max_words else count()
            # pylint: disable=not-callable
            should_stop = torch.tensor(False, device=device).repeat(batch_size)
        else:
            assert reports is not None, 'Cant pass free=False and reports=None'
            actual_max_words = reports.size()[-1]
            words_iterator = range(actual_max_words)
            should_stop = None


        for word_j in words_iterator:
            # Prepare input
            input_t = self.word_embeddings(next_words_indices)
            input_t = self.word_embeddings_bn(input_t)
            # shape: batch_size, embedding_size

            # Pass thru Word LSTM
            h_state_t, c_state_t = self.word_lstm(input_t, (h_state_t, c_state_t))
            # shapes: batch_size, hidden_size

            # Predict words
            prediction_t = self.word_fc(h_state_t) # shape: batch_size, vocab_size
            words_out.append(prediction_t)

            # Decide if stop
            if free:
                is_end_prediction = prediction_t.argmax(dim=-1) == END_IDX # shape: batch_size
                should_stop |= is_end_prediction

                if should_stop.all():
                    break

            # Get input for next step
            if teacher_forcing:
                next_words_indices = reports[:, sentence_i, word_j]
                # shape: batch_size
            else:
                _, next_words_indices = prediction_t.max(dim=1)
                # shape: batch_size

        words_out = torch.stack(words_out, dim=1)
        # shape: batch_size, max_n_words, vocab_size

        return words_out
