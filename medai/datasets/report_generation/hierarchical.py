import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

from medai.datasets.common import BatchRGItems
from medai.datasets.common.sentences2organs import SentenceToOrgans
from medai.datasets.common.sentences2embeddings import SentenceToEmbeddings
from medai.utils.nlp import split_sentences_and_pad


def create_hierarchical_dataloader(dataset, include_masks=False,
                                   include_sentence_emb=False, **kwargs):
    """Creates a dataloader from a images-report dataset with hierarchical sequences.

    Outputted reports have shape (batch_size, n_sentences, n_words)
    """
    if include_masks:
        sentence_to_organs = SentenceToOrgans(dataset)
    if include_sentence_emb:
        sentence_to_embeddings = SentenceToEmbeddings(dataset)

    def _collate_fn(batch_tuples):
        images = []
        reports = []
        masks = []
        sentence_embeddings = []
        report_fnames = []
        max_sentence_len = -1
        max_n_sentences = -1

        # Grab images and reports
        for tup in batch_tuples:
            # Collate images and filename
            images.append(tup.image)
            report_fnames.append(tup.report_fname)

            # Collate report
            report = tup.report # shape(list): n_words
            report = split_sentences_and_pad(report)
            # shape: n_sentences, n_words

            max_sentence_len = max(max_sentence_len, report.size(-1))
            max_n_sentences = max(max_n_sentences, report.size(0))

            reports.append(report)

            if include_masks:
                # Get masks for each sentence of the report
                image_masks = tup.masks # shape: n_organs, height, width
                sample_mask = torch.stack([
                    sentence_to_organs.get_mask_for_sentence(sentence, image_masks) # height, width
                    for sentence in report
                ])
                # shape: n_sentences, height, width

                masks.append(sample_mask)

            if include_sentence_emb:
                emb = torch.stack([
                    sentence_to_embeddings[sentence]
                    for sentence in report
                ])
                # shape: n_sentences, embedding_size

                sentence_embeddings.append(emb)

        # Pad reports to the max_sentence_len across all reports
        padded_reports = [
            pad(report, (0, max_sentence_len - report.size(-1)))
            if report.size(-1) < max_sentence_len else report
            for report in reports
        ]

        images = torch.stack(images)
        # shape: batch_size, channels, height, width

        reports = pad_sequence(padded_reports, batch_first=True)
        # shape: batch_size, n_sentences, n_words

        if include_masks:
            # Pad masks (n_sentences dimension)
            masks = [
                pad(mask, (0, 0, 0, 0, 0, max_n_sentences - mask.size(0)))
                if mask.size(0) < max_n_sentences else mask
                for mask in masks
            ]
            masks = torch.stack(masks, dim=0)
            # shape: batch_size, n_sentences, height, width
        else:
            masks = None

        if include_sentence_emb:
            # Pad sentence embeddings with 0s
            sentence_embeddings = [
                pad(emb, (0, 0, 0, max_n_sentences - emb.size(0)))
                if emb.size(0) < max_n_sentences else emb
                for emb in sentence_embeddings
            ]
            sentence_embeddings = torch.stack(sentence_embeddings, dim=0)
            # shape: batch_size, n_sentences, emb_size
        else:
            sentence_embeddings = None

        # Compute stops
        stops = [torch.zeros(report.size(0)) for report in padded_reports]
        stops = pad_sequence(stops, batch_first=True, padding_value=1)
        # shape: batch_size, n_sentences

        return BatchRGItems(
            images=images,
            reports=reports,
            stops=stops,
            report_fnames=report_fnames,
            masks=masks,
            sentence_embeddings=sentence_embeddings,
        )

    return DataLoader(dataset, collate_fn=_collate_fn, **kwargs)
