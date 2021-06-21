import torch
from torch import nn


class CoAttention(nn.Module):
    def __init__(self,
                 hidden_size=512,
                 embeddings_size=512,
                 visual_size=1024,
                 context_size=512,
                 ):
        super().__init__()

        # Visual stuff
        _internal_size = visual_size
        self.W_v_h = nn.Linear(hidden_size, _internal_size)
        self.W_v = nn.Linear(visual_size, _internal_size)
        self.W_v_att = nn.Linear(_internal_size, 1)

        self.visual_flatten = nn.Flatten(start_dim=1, end_dim=2)

        # Tag stuff
        _internal_size = embeddings_size
        self.W_a_h = nn.Linear(hidden_size, _internal_size)
        self.W_a = nn.Linear(embeddings_size, _internal_size)
        self.W_a_att = nn.Linear(_internal_size, 1)

        # Common stuff
        self.softmax = nn.Softmax(dim=1)
        self.W_fc = nn.Linear(visual_size + embeddings_size, context_size)

    def forward(self, local_features, tags_embeddings, prev_hidden_state):
        # shapes:
        # local_features: bs, visual_size, f-height, f-width
        # tags_embeddings: bs, n_tags, embeddings_size
        # prev_hidden_state: bs, hidden_size

        ### Visual attention
        local_features = local_features.permute(0, 2, 3, 1) # shape: bs, f-h, f-w, visual_size
        local_features = self.visual_flatten(local_features) # shape: bs, n_pixels, visual_size

        visual_internal = torch.tanh(
            self.W_v_h(prev_hidden_state).unsqueeze(1) # shape: bs, 1, visual_size
            +
            self.W_v(local_features) # shape: bs, n_pixels, visual_size
        )
        # shape: bs, n_pixels, visual_size

        visual_internal = self.W_v_att(visual_internal) # shape: bs, n_pixels, 1
        visual_internal = visual_internal.squeeze(-1) # shape: bs, n_pixels

        alpha_v = self.softmax(visual_internal)
        # shape: bs, n_pixels

        visual_out = local_features * alpha_v.unsqueeze(-1) # shape: bs, n_pixels, visual_size
        visual_out = visual_out.sum(dim=1) # shape: bs, visual_size


        ### Tags attention
        tags_internal = torch.tanh(
            self.W_a_h(prev_hidden_state).unsqueeze(1) # shape: bs, 1, embedding_size
            +
            self.W_a(tags_embeddings) # shape: bs, n_tags, embedding_size
        )
        # shape: bs, n_tags, embedding_size

        tags_internal = self.W_a_att(tags_internal) # shape: bs, n_tags, 1
        tags_internal = tags_internal.squeeze(-1) # shape: bs, n_tags

        alpha_a = self.softmax(tags_internal) # shape: bs, n_tags

        tags_out = tags_embeddings * alpha_a.unsqueeze(-1) # shape: bs, n_tags, embedding_size
        tags_out = tags_out.sum(dim=1) # shape: bs, embedding_size


        ### Merge into a context
        context = torch.cat((visual_out, tags_out), dim=1) # shape: bs, visual_size + embedding_size
        context = self.W_fc(context) # shape: bs, context_size

        return context, alpha_v, alpha_a
