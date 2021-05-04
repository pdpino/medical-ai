from torch import nn

class AttentionTwoLayers(nn.Module):
    def __init__(self, features_size, lstm_size, internal_size=100,
                 double_bias=True):
        super().__init__()

        self.visual_fc = nn.Linear(features_size, internal_size)
        self.state_fc = nn.Linear(lstm_size, internal_size, bias=double_bias)

        self.transition = nn.Tanh()

        self.last_fc = nn.Linear(internal_size, 1)

        self.softmax = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Softmax(dim=-1),
        )

    def forward(self, features, h_state):
        # features shape: batch_size, n_features, height, width
        # h_state shape: batch_size, lstm_size

        features_reshape = features.permute(0, 2, 3, 1)
        features_out = self.visual_fc(features_reshape)
        # shape: batch_size, height, width, internal_size

        h_state_out = self.state_fc(h_state)
        # shape: batch_size, internal_size

        out = features_out + h_state_out.unsqueeze(1).unsqueeze(1)
        # shape: batch_size, height, width, internal_size

        # Activation
        out = self.transition(out)
        # shape: batch_size, height, width, internal_size

        out = self.last_fc(out) # shape: batch_size, height, width, 1
        out = out.squeeze(-1)
        batch_size, height, width = out.size()
        # shape: batch_size, height, width

        scores = self.softmax(out).view(batch_size, height, width)
        # shape: batch_size, height, width

        weigthed_features = features * scores.unsqueeze(1)
        # shape: batch_size, n_features, height, width

        weigthed_features = weigthed_features.sum(dim=-1).sum(dim=-1)
        # shape: batch_size, n_features

        return weigthed_features, scores
