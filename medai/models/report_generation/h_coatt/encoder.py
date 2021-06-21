import torch
from torch import nn
from torch.nn import functional as F

from medai.models.common import (
    get_adaptive_pooling_layer,
    load_imagenet_model,
)

def build_linear_layers(input_size, output_size, layers=None):
    if not layers or len(layers) == 0:
        return nn.Linear(input_size, output_size)

    def _build_layers(input_size, layers_def):
        layers = []

        current_size = input_size
        for layer_size in layers_def:
            layers.extend([
                nn.Linear(current_size, layer_size),
                nn.ReLU(),
                nn.Dropout(),
            ])
            current_size = layer_size

        return layers


    return nn.Sequential(
        *_build_layers(input_size, layers),
        nn.Linear(layers[-1], output_size),
    )

class HCoAttEncoder(nn.Module):
    def __init__(self,
                 cnn_name='densenet-121',
                 imagenet=True, dropout=0,
                 n_tags=210,
                 embedding_size=512,
                 k_top_tags=10,
                 mlc_layers=None,
                 ):
        super().__init__()

        self.features, self.features_size = load_imagenet_model(
            cnn_name,
            imagenet=imagenet,
            dropout=dropout,
        )
        self.global_pool = get_adaptive_pooling_layer('avg')
        self.classifier = build_linear_layers(self.features_size, n_tags, mlc_layers)

        self.tag_embedding = nn.Embedding(n_tags, embedding_size)

        self.k_top_tags = k_top_tags

    def forward(self, images):
        # images shape: batch_size, 3, height, width

        local_features = self.features(images)
        # shape: bs, features_size, f-height, f-width

        local_features = F.relu(local_features)
        # shape: same

        global_features = self.global_pool(local_features)
        # shape: bs, features_size

        tags = self.classifier(global_features)
        # shape: bs, n_tags

        # Choose topk tags
        unused_top_values, top_tags_idxs = torch.topk(tags, self.k_top_tags)

        # Embed tags
        tags_embeddings = self.tag_embedding(top_tags_idxs)
        # shape: bs, k_top_tags, embedding_size

        return local_features, tags, tags_embeddings
