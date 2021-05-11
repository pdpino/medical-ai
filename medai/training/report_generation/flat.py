import torch
from torch import nn

from medai.utils.nlp import PAD_IDX, END_IDX


def _clean_gen_reports(gen_words):
    """Cleans the generated reports.

    Args:
        gen_words -- tensor of shape batch_size, n_words, vocab_size
    Returns:
        list of lists with reports, shape: batch_size, n_words_per_report
        (with token indexes, not words).
    """
    assert gen_words.ndim == 3

    gen_words = gen_words.argmax(-1)
    # shape: bs, n_words

    clean_reports = []
    for report in gen_words:
        # report shape: n_words

        # Remove PAD_IDX
        report = report[(report != PAD_IDX)]
        # shape: n_useful_words

        # Find END token
        end_position, = (report == END_IDX).nonzero(as_tuple=True)
        end_position = len(report) if len(end_position) == 0 else end_position[0].item()
        # int indicating END position

        # Keep only before the END token
        report = report[:end_position]
        # shape: n_clean_words

        clean_reports.append(report.tolist())

    return clean_reports


def _clean_gt_reports(gt_reports):
    """Cleans the generated reports.

    Args:
        gt_reports -- tensor of shape batch_size, n_words
    Returns:
        list of lists with reports, shape: batch_size, n_words_per_report
        (with token indexes, not words).
    """
    clean_reports = []
    for report in gt_reports:
        # report shape: n_words
        report = report[(report != PAD_IDX) & (report != END_IDX)].tolist()

        clean_reports.append(report)

    return clean_reports


def get_step_fn_flat(model, optimizer=None, training=True, free=False,
                     device='cuda', max_words=200, **unused_kwargs):
    """Creates a step function for an Engine."""
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    assert not (free and training), 'Cannot set training=True and free=True'

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
        # shape: batch_size, n_words, vocab_size

        if not free:
            # Compute word classification loss
            loss = loss_fn(generated_words.permute(0, 2, 1), reports)

            batch_loss = loss.item()
        else:
            batch_loss = -1

        if training:
            loss.backward()
            optimizer.step()

        generated_words = generated_words.detach()

        return {
            'loss': batch_loss,
            'flat_clean_reports_gen': _clean_gen_reports(generated_words),
            'flat_clean_reports_gt': _clean_gt_reports(reports),
        }

    return step_fn
