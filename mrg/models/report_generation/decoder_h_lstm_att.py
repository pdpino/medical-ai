import torch
from torch import nn

from mrg.models.report_generation.att_2layer import AttentionTwoLayers
from mrg.utils.nlp import PAD_IDX, START_IDX, END_OF_SENTENCE_IDX


class HierarchicalLSTMAttDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size,
                 features_size,
                 teacher_forcing=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.teacher_forcing = teacher_forcing
        self.start_idx = torch.tensor(START_IDX)

        n_features, height, width = features_size

        # Sentence LSTM
        self.attention = AttentionTwoLayers(features_size, hidden_size)
        self.sentence_lstm = nn.LSTMCell(n_features, hidden_size)
        self.stop_control = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        
        # Word LSTM
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_IDX)
        self.word_lstm = nn.LSTMCell(embedding_size, hidden_size)
        self.word_fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, unused_FIXME, reports=None):
        # features shape: batch_size, n_features, height, width
        batch_size = features.size()[0]
        device = features.device

        # Build initial state # TODO: use global features to initialize?
        initial_h_state = torch.zeros(batch_size, self.hidden_size).to(device)
        initial_c_state = torch.zeros(batch_size, self.hidden_size).to(device)
        state = (initial_h_state, initial_c_state)

        # Build initial input
        att_features, att_scores = self.attention(features, initial_h_state)
            # att_features shape: batch_size, n_features
            # att_scores features: batch_size, height, width
        sentence_input_t = att_features

        # Decide teacher forcing
        teacher_forcing = self.teacher_forcing and self.training and reports is not None
        
        # Iterate over sentences
        seq_out = []
        stops_out = []
        scores_out = []

        if reports is not None:
            batch_size, max_n_sentences, max_n_words = reports.size()
        else:
            raise NotImplementedError
        
        # TODO: don't use max_n_sentences if not teacher_forcing?
        # i.e notice when stop is True (> 0.5, or other threshold)
        for sentence_i in range(max_n_sentences):
            # Pass thru LSTM
            state = self.sentence_lstm(sentence_input_t, state)
            h_t, c_t = state
            # shapes: batch_size, hidden_size

            # Decide stop
            stop = self.stop_control(h_t).squeeze(-1) # shape: batch_size
            stops_out.append(stop)
            if not teacher_forcing:
                pass # TODO: decide to stop or not

            # Generate topic vector
            topic = h_t
            
            # Generate sentence with topic
            words = self.generate_sentence(topic,
                                           max_n_words,
                                           sentence_i,
                                           reports=reports,
                                           teacher_forcing=teacher_forcing,
                                          )
            seq_out.append(words)
            
            # Get next input
            att_features, att_scores = self.attention(features, initial_h_state)
            sentence_input_t = att_features
            scores_out.append(att_scores)

        seq_out = torch.stack(seq_out, dim=1)
        # shape: batch_size, max_n_sentences, max_n_words, vocab_size

        stops_out = torch.stack(stops_out, dim=1)
        # shape: batch_size, max_n_sentences
        
        scores_out = torch.stack(scores_out, dim=1)
        # shape: batch_size, max_n_sentences, height, width

        return seq_out, stops_out, scores_out
    
    def generate_sentence(self, topic, max_n_words, sentence_i,
                          reports=None, teacher_forcing=True):
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
        
        # TODO: don't use max_n_words if not teacher_forcing?
        # i.e. notice when end_of_sentence_idx is generated
        for word_j in range(max_n_words):
            # Pass thru Word LSTM
            state = self.word_lstm(input_t, state)
            h_t, c_t = state
            # shapes: batch_size, hidden_size
            
            # Predict words
            prediction_t = self.word_fc(h_t) # shape: batch_size, vocab_size
            words_out.append(prediction_t)
            
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