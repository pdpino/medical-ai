import torch
from torch import nn

from medai.utils.nlp import START_IDX

class SentenceLSTM(nn.Module):
    def __init__(self,
                 version='v1',
                 embed_size=512,
                 hidden_size=512,
                 num_layers=1,
                 dropout=0.3,
                 momentum=0.1,
                 ):
        super(SentenceLSTM, self).__init__()
        self.version = version

        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout)

        self.W_t_h = nn.Linear(in_features=hidden_size,
                               out_features=embed_size,
                               bias=True)
        self.bn_t_h = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_t_ctx = nn.Linear(in_features=embed_size,
                                 out_features=embed_size,
                                 bias=True)
        self.bn_t_ctx = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_stop_s_1 = nn.Linear(in_features=hidden_size,
                                    out_features=embed_size,
                                    bias=True)
        self.bn_stop_s_1 = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_stop_s = nn.Linear(in_features=hidden_size,
                                  out_features=embed_size,
                                  bias=True)
        self.bn_stop_s = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_stop = nn.Linear(in_features=embed_size,
                                out_features=2,
                                bias=True)
        self.bn_stop = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_topic = nn.Linear(in_features=embed_size,
                                 out_features=embed_size,
                                 bias=True)
        self.bn_topic = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.__init_weight()

    def __init_weight(self):
        self.W_t_h.weight.data.uniform_(-0.1, 0.1)
        self.W_t_h.bias.data.fill_(0)

        self.W_t_ctx.weight.data.uniform_(-0.1, 0.1)
        self.W_t_ctx.bias.data.fill_(0)

        self.W_stop_s_1.weight.data.uniform_(-0.1, 0.1)
        self.W_stop_s_1.bias.data.fill_(0)

        self.W_stop_s.weight.data.uniform_(-0.1, 0.1)
        self.W_stop_s.bias.data.fill_(0)

        self.W_stop.weight.data.uniform_(-0.1, 0.1)
        self.W_stop.bias.data.fill_(0)

        self.W_topic.weight.data.uniform_(-0.1, 0.1)
        self.W_topic.bias.data.fill_(0)

    def forward(self, ctx, prev_hidden_state, states=None) -> object:
        """
        :rtype: object
        """
        if self.version == 'v1':
            return self.v1(ctx, prev_hidden_state, states)
        elif self.version == 'v2':
            return self.v2(ctx, prev_hidden_state, states)
        elif self.version == 'v3':
            return self.v3(ctx, prev_hidden_state, states)

    def v1(self, ctx, prev_hidden_state, states=None):
        """
        v1 (only training)
        :param ctx:
        :param prev_hidden_state:
        :param states:
        :return:
        """
        ctx = ctx.unsqueeze(1)
        hidden_state, states = self.lstm(ctx, states)
        topic = self.W_topic(self.sigmoid(self.bn_t_h(self.W_t_h(hidden_state))
                                          + self.bn_t_ctx(self.W_t_ctx(ctx))))
        p_stop = self.W_stop(self.sigmoid(self.bn_stop_s_1(self.W_stop_s_1(prev_hidden_state))
                                          + self.bn_stop_s(self.W_stop_s(hidden_state))))
        return topic, p_stop, hidden_state, states

    def v2(self, ctx, prev_hidden_state, states=None):
        """
        v2
        :rtype: object
        """
        ctx = ctx.unsqueeze(1)
        hidden_state, states = self.lstm(ctx, states)
        topic = self.bn_topic(self.W_topic(self.tanh(self.bn_t_h(self.W_t_h(hidden_state)
                                                                 + self.W_t_ctx(ctx)))))
        p_stop = self.bn_stop(self.W_stop(self.tanh(self.bn_stop_s(
            self.W_stop_s_1(prev_hidden_state) + self.W_stop_s(hidden_state),
        ))))
        return topic, p_stop, hidden_state, states

    def v3(self, ctx, prev_hidden_state, states=None):
        """
        v3
        :rtype: object
        """
        ctx = ctx.unsqueeze(1)
        hidden_state, states = self.lstm(ctx, states)
        topic = self.W_topic(self.tanh(self.W_t_h(hidden_state) + self.W_t_ctx(ctx)))
        p_stop = self.W_stop(self.tanh(
            self.W_stop_s_1(prev_hidden_state) + self.W_stop_s(hidden_state),
        ))
        return topic, p_stop, hidden_state, states


class WordLSTM(nn.Module):
    def __init__(self,
                 embed_size,
                 hidden_size,
                 vocab_size,
                 num_layers,
                 n_max=50):
        super(WordLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.__init_weights()
        self.n_max = n_max
        self.vocab_size = vocab_size

    def __init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, topic, captions=None):
        if self.training:
            return self.forward_train(topic, captions)
        return self.forward_test(topic)

    def forward_train(self, topic_vec, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((topic_vec, embeddings), 1)
        hidden, _ = self.lstm(embeddings)
        outputs = self.linear(hidden[:, -1, :])
        return outputs

    def forward_test(self, topic):
        batch_size = topic.size(0)
        start_tokens = torch.zeros(batch_size, 1, device=topic.device).fill_(START_IDX).long()

        sampled_ids = torch.zeros(batch_size, self.n_max, device=topic.device)
        sampled_ids[:, 0] = START_IDX
        predicted = start_tokens
        embeddings = topic

        for i in range(1, self.n_max):
            predicted = self.embed(predicted)
            embeddings = torch.cat([embeddings, predicted], dim=1)
            hidden_states, _ = self.lstm(embeddings)
            hidden_states = hidden_states[:, -1, :]
            outputs = self.linear(hidden_states)
            predicted = torch.max(outputs, 1)[1]
            sampled_ids[:, i] = predicted
            predicted = predicted.unsqueeze(1)
        return sampled_ids
