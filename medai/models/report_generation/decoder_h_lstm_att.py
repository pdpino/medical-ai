from itertools import count
import torch
from torch import nn

from medai.models.report_generation.att_2layer import AttentionTwoLayers
from medai.models.report_generation.att_no_att import NoAttention
from medai.utils.nlp import PAD_IDX, START_IDX, END_OF_SENTENCE_IDX, END_IDX


def h_lstm_wrapper(attention=True):
    def constructor(*args, **kwargs):
        return HierarchicalLSTMAttDecoder(*args, attention=attention, **kwargs)
    return constructor


class HierarchicalLSTMAttDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size,
                 features_size, teacher_forcing=True, stop_threshold=0.5,
                 attention=True, **kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.teacher_forcing = teacher_forcing
        self.start_idx = torch.tensor(START_IDX)
        self.stop_threshold = stop_threshold

        # Attention input
        self._use_attention = attention
        if attention:
            self.attention_layer = AttentionTwoLayers(features_size, hidden_size)
        else:
            self.attention_layer = NoAttention(reduction='mean')

        # Sentence LSTM
        self.sentence_lstm = nn.LSTMCell(features_size, hidden_size)
        self.stop_control = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        
        # Word LSTM
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_IDX)
        self.word_lstm = nn.LSTMCell(embedding_size, hidden_size)
        self.word_fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, reports=None, free=False, max_sentences=100, max_words=1000):
        # features shape: batch_size, features_size, height, width
        batch_size = features.size()[0]
        device = features.device

        # Build initial state # TODO: use global features to initialize?
        initial_h_state = torch.zeros(batch_size, self.hidden_size).to(device)
        initial_c_state = torch.zeros(batch_size, self.hidden_size).to(device)
        state = (initial_h_state, initial_c_state)

        # Build initial input
        att_features, att_scores = self.attention_layer(features, initial_h_state)
            # att_features shape: batch_size, features_size
            # att_scores features: batch_size, height, width (or None)
        sentence_input_t = att_features

        # Decide teacher forcing
        teacher_forcing = self.teacher_forcing \
            and self.training \
            and reports is not None \
            and not free

        # Set iteration maximum
        if free:
            sentences_iterator = range(max_sentences) if max_sentences else count()
            should_stop = torch.tensor(False).to(device).repeat(batch_size)
        else:
            assert reports is not None, 'Cant pass free=False and reports=None'
            actual_n_sentences = reports.size()[1]
            sentences_iterator = range(actual_n_sentences)
            should_stop = None

        # Iterate over sentences
        seq_out = []
        stops_out = []
        scores_out = [] # REVIEW: should add the first scores??

        for sentence_i in sentences_iterator:
            # Pass thru LSTM
            state = self.sentence_lstm(sentence_input_t, state)
            h_t, c_t = state
            # shapes: batch_size, hidden_size

            # Generate topic vector
            topic = h_t
            
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
            stop = self.stop_control(h_t).squeeze(-1) # shape: batch_size
            stops_out.append(stop)
            if free:
                should_stop |= (stop >= self.stop_threshold)

                if should_stop.all():
                    break

            # Get next input
            att_features, att_scores = self.attention_layer(features, h_t)
            sentence_input_t = att_features

            if self._use_attention:
                scores_out.append(att_scores)

        seq_out = torch.stack(seq_out, dim=1)
        # shape: batch_size, n_sentences, max_n_words, vocab_size

        stops_out = torch.stack(stops_out, dim=1)
        # shape: batch_size, n_sentences
        
        if self._use_attention and len(scores_out) > 0:
            scores_out = torch.stack(scores_out, dim=1)
        else:
            scores_out = None
        # shape: batch_size, n_sentences, height, width (or None)

        return seq_out, stops_out, scores_out
    
    def generate_sentence(self, topic, sentence_i,
                          reports=None, teacher_forcing=True, free=False, max_words=1000):
        # REVIEW: re-use this logic with a WordDecoder ???

        # topic shape: batch_size, hidden_size
        batch_size = topic.size()[0]
        device = topic.device
        
        # Build initial state
        initial_h_state = topic
        initial_c_state = torch.zeros(batch_size, self.hidden_size).to(device)
        state = (initial_h_state, initial_c_state)
        
        # Build initial input
        start_idx = self.start_idx.to(device).repeat(batch_size)
        input_t = self.word_embeddings(start_idx)

        words_out = []
        

        # Set iteration maximum
        if free:
            words_iterator = range(max_words) if max_words else count()
            should_stop = torch.tensor(False).to(device).repeat(batch_size)
        else:
            assert reports is not None, 'Cant pass free=False and reports=None'
            actual_max_words = reports.size()[-1]
            words_iterator = range(actual_max_words)
            should_stop = None


        for word_j in words_iterator:
            # Pass thru Word LSTM
            state = self.word_lstm(input_t, state)
            h_t, c_t = state
            # shapes: batch_size, hidden_size
            
            # Predict words
            prediction_t = self.word_fc(h_t) # shape: batch_size, vocab_size
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
            
            input_t = self.word_embeddings(next_words_indices)
            # shape: batch_size, embedding_size

        words_out = torch.stack(words_out, dim=1)
        # shape: batch_size, max_n_words, vocab_size
        
        return words_out