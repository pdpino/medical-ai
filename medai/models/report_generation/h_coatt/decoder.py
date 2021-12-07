from itertools import count
import torch
from torch import nn
from torch.nn import functional as F

from medai.models.report_generation.h_coatt.att import CoAttention
from medai.models.report_generation.word_embedding import create_word_embedding
from medai.utils.nlp import END_IDX


class TopicGenerator(nn.Module):
    def __init__(self, hidden_size, context_size, topic_size):
        super().__init__()

        self.W_t_h = nn.Linear(hidden_size, topic_size)
        self.W_t_ctx = nn.Linear(context_size, topic_size)

    def forward(self, hidden_state, context):
        # shapes:
        # hidden_state: bs, hidden_size
        # context: bs, context_size

        topic = torch.tanh(
            self.W_t_h(hidden_state) # shape: bs, topic_size
            +
            self.W_t_ctx(context) # shape: bs, topic_size
        )
        # shape: bs, internal_size

        return topic


class StopControl(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        _internal_size = hidden_size
        self.W_stop_s_1 = nn.Linear(hidden_size, _internal_size)
        self.W_stop_s = nn.Linear(hidden_size, _internal_size)
        self.W_stop = nn.Linear(_internal_size, 1)

    def forward(self, hidden_state, prev_hidden_state):
        # shapes: bs, hidden_size

        stop = torch.tanh(
            self.W_stop_s(hidden_state) # shape: bs, internal_size
            +
            self.W_stop_s_1(prev_hidden_state) # shape: bs, internal_size
        )
        # shape: bs, internal_size

        stop = self.W_stop(stop)
        # shape: bs, 1

        stop = torch.sigmoid(stop).squeeze(-1)
        # shape: batch_size

        return stop


class HCoAttDecoder(nn.Module):
    def __init__(self,
                 vocab,
                 features_size=1024,
                 hidden_size=512,
                 tag_embedding_size=512,
                 word_embedding_size=512,
                 embedding_kwargs={},
                 context_size=512,
                 stop_threshold=0.5,
                 topic_size=512,
                 ):
        super().__init__()
        if topic_size != word_embedding_size:
            sizes = f'got topic={topic_size} vs word={word_embedding_size}'
            raise Exception(f'Topic size must be equal to word-emb-size {sizes}')

        self.hidden_size = hidden_size
        self.stop_threshold = stop_threshold
        self.teacher_forcing = True

        # Attention
        self.co_attention = CoAttention(
            hidden_size=hidden_size,
            embeddings_size=tag_embedding_size,
            visual_size=features_size,
            context_size=context_size,
        )

        # Sentence LSTM
        self.sentence_lstm = nn.LSTMCell(context_size, hidden_size)
        self.stop_control = StopControl(hidden_size)
        self.topic_generator = TopicGenerator(hidden_size, context_size, topic_size)

        # Word LSTM
        self.word_embeddings, _ = create_word_embedding(
            vocab,
            word_embedding_size,
            **embedding_kwargs,
        )
        self.word_lstm = nn.LSTMCell(word_embedding_size, hidden_size)
        self.word_fc = nn.Linear(hidden_size, len(vocab))

        self.dropout_out = 0
        self.dropout_recursive = 0


    def forward(self, local_features, tags_embeddings, reports=None,
                free=False, max_sentences=50, max_words=200):
        # shapes:
        # local_features: bs, features_size, f-height, f-width
        # tags_embeddings: bs, k_top_tags, embedding_size
        # reports: bs, n_sentences, n_words
        batch_size = local_features.size(0)
        device = local_features.device

        # Build initial state
        h_state_t = torch.zeros(batch_size, self.hidden_size, device=device)
        h_state_prev = h_state_t
        c_state_t = torch.zeros(batch_size, self.hidden_size, device=device)

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
            actual_n_sentences = reports.size(1)
            sentences_iterator = range(actual_n_sentences)
            should_stop = None

        # Iterate over sentences
        seq_out = []
        stops_out = []
        scores_out_visual = []
        scores_out_tags = []
        topics_out = []

        for sentence_i in sentences_iterator:
            context, att_scores_visual, att_scores_tags = self.co_attention(
                local_features, tags_embeddings, h_state_t,
            )
            # shapes
            # context: bs, context_size
            # att_scores_visual: bs, n_pixels
            # att_scores_tags: bs, n_top_tags

            # Save att-scores
            scores_out_visual.append(att_scores_visual)
            scores_out_tags.append(att_scores_tags)

            # Pass thru LSTM
            h_state_t, c_state_t = self.sentence_lstm(context, (h_state_t, c_state_t))
            # shapes: batch_size, hidden_size

            # Generate topic vector
            topic = self.topic_generator(h_state_t, context)
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
            stop = self.stop_control(h_state_t, h_state_prev) # shape: bs
            stops_out.append(stop)
            if free:
                should_stop |= (stop >= self.stop_threshold)

                if should_stop.all():
                    break

            # Save
            h_state_prev = h_state_t

        seq_out = torch.stack(seq_out, dim=1)
        # shape: batch_size, n_sentences, max_n_words, vocab_size

        stops_out = torch.stack(stops_out, dim=1)
        # shape: batch_size, n_sentences

        scores_out_visual = torch.stack(scores_out_visual, dim=1) # shape: bs, n_sentences, n_pixels
        scores_out_tags = torch.stack(scores_out_tags, dim=1) # shape: bs, n_sentences, n_top_tags

        topics_out = torch.stack(topics_out, dim=1)
        # shape: batch_size, n_sentences, hidden_size

        return seq_out, stops_out, scores_out_visual, scores_out_tags, topics_out


    def generate_sentence(self, topic, sentence_i,
                          reports=None, teacher_forcing=True, free=False, max_words=1000):
        # topic shape: batch_size, topic_size
        batch_size = topic.size(0)
        device = topic.device

        # Build initial state
        h_state_t = torch.zeros(batch_size, self.hidden_size, device=device)
        c_state_t = torch.zeros(batch_size, self.hidden_size, device=device)

        # Pass topic first
        # NOTICE: topic_size must be == word_embedding_size
        h_state_t, c_state_t = self.word_lstm(topic, (h_state_t, c_state_t))

        # Build initial input
        next_words_indices = None
        input_t = topic

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
            if word_j != 0:
                # Prepare input
                input_t = self.word_embeddings(next_words_indices)
                # shape: batch_size, embedding_size

                # For word_j == 0 the input_t is the topic

            # Pass thru Word LSTM
            h_state_t, c_state_t = self.word_lstm(input_t, (h_state_t, c_state_t))
            # shapes: batch_size, hidden_size

            # Pass thru out-dropout, if any
            out_h_t = h_state_t
            if self.dropout_out:
                out_h_t = F.dropout(out_h_t, self.dropout_out, training=self.training)

            # Predict words
            prediction_t = self.word_fc(out_h_t) # shape: batch_size, vocab_size
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

            # Apply recursive dropout
            if self.dropout_recursive:
                h_state_t = F.dropout(h_state_t, self.dropout_recursive, training=self.training)
                c_state_t = F.dropout(c_state_t, self.dropout_recursive, training=self.training)

        words_out = torch.stack(words_out, dim=1)
        # shape: batch_size, max_n_words, vocab_size

        return words_out
