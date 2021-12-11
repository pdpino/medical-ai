import torch
from torch import nn
from torch.nn.functional import pad, one_hot
from torch.nn.utils.rnn import pad_sequence

def _cosine_similarity(a, b, eps=1e-8):
    """Cosine similarity calculation taken from:

    https://stackoverflow.com/a/58144658/9951939.
    """
    # a shape: n, features
    # b shape: m, features
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    # shape: n, m

    return sim_mt


def _cosine_closest(features, feats_database, eps=1e-8):
    # a shape: batch_size, features
    # b shape: dataset_size, features
    # assert features.size(1) == feats_database.size(1)

    # Compute cosine similarity (largest --> closest)
    cos_sim = _cosine_similarity(features, feats_database, eps=eps)
    # shape: batch_size, dataset_size

    _, most_similar = cos_sim.max(dim=-1)
    # shape: batch_size

    return most_similar

def _euclidean_closest(features, feats_database):
    distances = torch.cdist(features, feats_database)
    # shape: batch_size, dataset_size

    _, closest = distances.min(dim=-1)
    # shape: batch_size
    return closest


_DISTANCES = {
    'euc': _euclidean_closest,
    'cos': _cosine_closest,
}

AVAILABLE_DISTANCES = list(_DISTANCES)

class MostSimilarImage(nn.Module):
    """Returns the report from the most similar image.

    NOTE: loads all features in memory, in the fit() method
    """
    def __init__(self, cnn, vocab, distance):
        super().__init__()

        if distance not in _DISTANCES:
            raise Exception(f'1-nn distance not available {distance}')

        self.distance_fn = _DISTANCES[distance]

        self.cnn = cnn

        self.global_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
        )
        self.vocab_size = len(vocab)

        self.all_features = None
        self.all_reports = None

    def fit(self, dataloader, device='cuda'):
        all_features = []
        all_reports = []

        for batch in iter(dataloader):
            images = batch.images.to(device)
            features = self.images_to_features(images)
            # shape: batch_size, features_size

            all_features.append(features)

            reports = batch.reports.to(device)
            all_reports.extend(reports)

        self.all_features = torch.cat(all_features, dim=0)
        # shape: dataset_size, features_size

        self.all_reports = all_reports
        # shape: dataset_size, n_words

        assert len(self.all_reports) == self.all_features.size(0)


    def images_to_features(self, images):
        with torch.no_grad():
            features = self.global_pool(self.cnn.features(images))
        return features


    def forward(self, images, reports=None, free=False, **unused_kwargs):
        features = self.images_to_features(images)
        # shape: batch_size, features_size

        closest = self.distance_fn(features, self.all_features)
        # shape: batch_size

        output_reports = [
            self.all_reports[index.item()]
            for index in closest
        ]
        # list shape: batch_size, n_words

        output_reports = pad_sequence(output_reports, batch_first=True)
        # tensor shape: batch_size, n_words

        if reports is not None and not free:
            n_words_target = reports.size(1)
            n_words_current = output_reports.size(1)

            if n_words_current > n_words_target:
                output_reports = output_reports[:, :n_words_target]
            elif n_words_current < n_words_target:
                missing = n_words_target - n_words_current
                output_reports = pad(output_reports, (0, missing))

            # shape: batch_size, n_words_target, vocab_size

        output_reports = one_hot(output_reports, num_classes=self.vocab_size).float()
        # shape: batch_size, n_words, vocab_size

        return (output_reports,)
