import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from medai.datasets.common import BatchItems
from medai.utils.nlp import END_IDX


def create_flat_dataloader(dataset, **kwargs):
    """Creates a dataloader from a images-report dataset, considering flat word sequences.

    Outputed reports have shape (batch_size, n_words)
    Adds END_TOKEN to the end of the sentences, and pads the output sequence.
    """
    def _collate_fn(batch_tuples):
        images = []
        reports = []
        filenames = []
        for tup in batch_tuples:
            images.append(tup.image)
            filenames.append(tup.filename)
            reports.append(torch.tensor(tup.report + [END_IDX]))

        images = torch.stack(images)
        reports = pad_sequence(reports, batch_first=True)
        return BatchItems(
            images=images,
            reports=reports,
            filenames=filenames,
        )

    return DataLoader(dataset, collate_fn=_collate_fn, **kwargs)


def get_step_fn_flat(model, optimizer=None, training=True, free=False,
                     device='cuda', max_words=200, **unused):
    """Creates a step function for an Engine."""
    loss_fn = nn.CrossEntropyLoss()

    assert not (free and training), 'Cant set training=True and free=True'

    def step_fn(engine, data_batch):
        # Images
        images = data_batch.images.to(device)
        # shape: batch_size, 3, height, width

        # Reports, as word ids
        reports = data_batch.reports.to(device).long()
        # shape: batch_size, max_sentence_len

        # Enable training
        model.train(training)
        torch.set_grad_enabled(training) # enable recording gradients

        # zero the parameter gradients
        if training:
            optimizer.zero_grad()

        # Pass thru the model
        output_tuple = model(images, reports, free=free, max_words=max_words)

        generated_words = output_tuple[0]
        _, _, vocab_size = generated_words.size()
        # shape: batch_size, n_sentences, vocab_size

        if not free:
            # Compute word classification loss
            loss = loss_fn(generated_words.view(-1, vocab_size), reports.view(-1))

            batch_loss = loss.item()
        else:
            batch_loss = -1

        if training:
            loss.backward()
            optimizer.step()

        _, flat_reports_gen = generated_words.max(dim=-1)
        return {
            'loss': batch_loss,
            'words_scores': generated_words,
            'flat_reports_gen': flat_reports_gen, # shape: batch_size, n_words
            'flat_reports': reports, # shape: batch_size, n_words
        }

    return step_fn