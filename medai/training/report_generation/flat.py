import torch
from torch import nn

from medai.utils.nlp import PAD_IDX


def get_step_fn_flat(model, optimizer=None, training=True, free=False,
                     device='cuda', max_words=200, **unused_kwargs):
    """Creates a step function for an Engine."""
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    assert not (free and training), 'Cant set training=True and free=True'

    def step_fn(unused_engine, data_batch):
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
        # shape: batch_size, n_words, vocab_size

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
