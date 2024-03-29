import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from medai.datasets.common import BatchRGItems
from medai.utils.nlp import END_IDX

def create_flat_dataloader(dataset, **kwargs):
    """Creates a dataloader from a images-report dataset, considering flat word sequences.

    Outputed reports have shape (batch_size, n_words)
    Adds END_TOKEN to the end of the sentences, and pads the output sequence.
    """
    def _collate_fn(batch_tuples):
        images = []
        reports = []
        report_fnames = []
        image_fnames = []
        for tup in batch_tuples:
            images.append(tup.image)
            report_fnames.append(tup.report_fname)
            image_fnames.append(tup.image_fname)
            reports.append(torch.tensor(tup.report + [END_IDX])) # pylint: disable=not-callable

        images = torch.stack(images)
        reports = pad_sequence(reports, batch_first=True)
        return BatchRGItems(
            images=images,
            reports=reports,
            report_fnames=report_fnames,
            image_fnames=image_fnames,
        )

    return DataLoader(dataset, collate_fn=_collate_fn, **kwargs)
