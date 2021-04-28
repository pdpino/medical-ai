import torch
from torch import nn
from torch.nn.functional import pad, one_hot
from torch.nn.utils.rnn import pad_sequence

def _extract_reports(dataset):
    return [
        # pylint: disable=not-callable
        torch.tensor(report['tokens_idxs'])
        for report in dataset.iter_reports_only()
    ]

class MostSimilarImage(nn.Module):
    """Returns the report from the most similar image.

    NOTE: loads all features in memory, in the fit() method
    """
    def __init__(self, cnn, vocab):
        super().__init__()

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
            all_features.append(features)

            reports = batch.reports.to(device)
            all_reports.extend(reports)

        self.all_features = torch.cat(all_features, dim=0)
        # shape: dataset_size, features_size

        self.all_reports = all_reports
        # shape: dataset_size, n_words


    def images_to_features(self, images):
        with torch.no_grad():
            features = self.cnn(images, features=True)
            features = self.global_pool(features)
        return features


    def forward(self, images, reports=None, free=False, **unused_kwargs):
        device = images.device

        features = self.images_to_features(images)
        # shape: batch_size, features_size

        distances = torch.cdist(features, self.all_features)
        # shape: batch_size, dataset_size

        _, closest = distances.min(dim=-1)
        # shape: batch_size

        output_reports = [
            self.all_reports[index.item()]
            for index in closest
        ]
        # list shape: batch_size, n_words

        output_reports = pad_sequence(output_reports, batch_first=True)
        # tensor shape: batch_size, n_words

        if reports is not None and not free:
            n_words_target = reports.size()[1]
            n_words_current = output_reports.size()[1]

            if n_words_current > n_words_target:
                output_reports = output_reports[:, :n_words_target]
            elif n_words_current < n_words_target:
                missing = n_words_target - n_words_current
                output_reports = pad(output_reports, (0, missing))

            # shape: batch_size, n_words_target, vocab_size

        output_reports = one_hot(output_reports, num_classes=self.vocab_size).float()
        # shape: batch_size, n_words, vocab_size

        output_reports = output_reports.to(device)

        return output_reports, distances
