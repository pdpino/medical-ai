import torch
from torch import nn


class CoAttention(nn.Module):
    def __init__(self,
                 version='v1',
                 embed_size=512,
                 hidden_size=512,
                 visual_size=2048,
                 k=10,
                 momentum=0.1):
        super(CoAttention, self).__init__()
        self.version = version
        self.W_v = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_h = nn.Linear(in_features=hidden_size, out_features=visual_size)
        self.bn_v_h = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_att = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v_att = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_a = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a = nn.BatchNorm1d(num_features=k, momentum=momentum)

        self.W_a_h = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_h = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_a_att = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_att = nn.BatchNorm1d(num_features=k, momentum=momentum)

        # self.W_fc = nn.Linear(in_features=visual_size, out_features=embed_size)  # for v3
        self.W_fc = nn.Linear(in_features=visual_size + hidden_size, out_features=embed_size)
        self.bn_fc = nn.BatchNorm1d(num_features=embed_size, momentum=momentum)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.__init_weight()

    def __init_weight(self):
        self.W_v.weight.data.uniform_(-0.1, 0.1)
        self.W_v.bias.data.fill_(0)

        self.W_v_h.weight.data.uniform_(-0.1, 0.1)
        self.W_v_h.bias.data.fill_(0)

        self.W_v_att.weight.data.uniform_(-0.1, 0.1)
        self.W_v_att.bias.data.fill_(0)

        self.W_a.weight.data.uniform_(-0.1, 0.1)
        self.W_a.bias.data.fill_(0)

        self.W_a_h.weight.data.uniform_(-0.1, 0.1)
        self.W_a_h.bias.data.fill_(0)

        self.W_a_att.weight.data.uniform_(-0.1, 0.1)
        self.W_a_att.bias.data.fill_(0)

        self.W_fc.weight.data.uniform_(-0.1, 0.1)
        self.W_fc.bias.data.fill_(0)

    def forward(self, avg_features, semantic_features, h_sent):
        if self.version == 'v1':
            return self.v1(avg_features, semantic_features, h_sent)
        elif self.version == 'v2':
            return self.v2(avg_features, semantic_features, h_sent)
        elif self.version == 'v3':
            return self.v3(avg_features, semantic_features, h_sent)
        elif self.version == 'v4':
            return self.v4(avg_features, semantic_features, h_sent)
        elif self.version == 'v5':
            return self.v5(avg_features, semantic_features, h_sent)

    def v1(self, avg_features, semantic_features, h_sent) -> object:
        """
        only training
        :rtype: object
        """
        W_v = self.bn_v(self.W_v(avg_features))
        W_v_h = self.bn_v_h(self.W_v_h(h_sent.squeeze(1)))

        alpha_v = self.softmax(self.bn_v_att(self.W_v_att(self.tanh(W_v + W_v_h))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.bn_a_h(self.W_a_h(h_sent))
        W_a = self.bn_a(self.W_a(semantic_features))
        alpha_a = self.softmax(self.bn_a_att(self.W_a_att(self.tanh(torch.add(W_a_h, W_a)))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v2(self, avg_features, semantic_features, h_sent) -> object:
        """
        no bn
        :rtype: object
        """
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(W_v + W_v_h)))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v3(self, avg_features, semantic_features, h_sent) -> object:
        """

        :rtype: object
        """
        W_v = self.bn_v(self.W_v(avg_features))
        W_v_h = self.bn_v_h(self.W_v_h(h_sent.squeeze(1)))

        alpha_v = self.softmax(self.W_v_att(self.tanh(W_v + W_v_h)))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.bn_a_h(self.W_a_h(h_sent))
        W_a = self.bn_a(self.W_a(semantic_features))
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v4(self, avg_features, semantic_features, h_sent):
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(torch.add(W_v, W_v_h))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v5(self, avg_features, semantic_features, h_sent):
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(self.bn_v(torch.add(W_v, W_v_h)))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(self.bn_a(torch.add(W_a_h, W_a)))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a
