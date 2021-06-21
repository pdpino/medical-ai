"""Custom step_fn for the CoAtt model.

This function is for the medai-based model (h-coatt).
"""
import logging
import torch
from torch import nn

from medai.training.report_generation.hierarchical import _flatten_gt_reports, _flatten_gen_reports
from medai.utils.nlp import PAD_IDX

LOGGER = logging.getLogger(__name__)


def _calculate_regularization(scores_visual, scores_tags):
    scores_visual = scores_visual.sum(dim=1) # shape: bs, n_pixels
    scores_visual = (1 - scores_visual) ** 2 # shape: bs, n_pixels
    scores_visual = scores_visual.mean(dim=-1) # shape: bs,

    scores_tags = scores_tags.sum(dim=1) # shape: bs, n_top_tags
    scores_tags = (1 - scores_tags) ** 2 # shape: bs, n_top_tags
    scores_tags = scores_tags.mean(dim=-1) # shape: bs,

    return (scores_visual + scores_tags).mean() # shape: 1


def get_step_fn_h_coatt(model, optimizer=None, training=True,
                        lambda_tag=1, lambda_stop=1, lambda_word=1, lambda_reg=1,
                        free=False,
                        device='cuda',
                        max_words=100, max_sentences=20,
                        **unused_kwargs,
                        ):
    """Return a step-fn for the CoAtt model.

    Notice the stops are inverted inside the co-att model (with respect to default medai behavior):
        1 means continue, 0 means stop.
    """
    assert not (free and training), 'Cannot set training=True and free=True'
    word_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    stop_loss_fn = nn.BCELoss()
    tag_loss_fn = nn.BCEWithLogitsLoss()
    # tag_loss_fn = nn.CrossEntropyLoss()

    LOGGER.info(
        'Using lambdas: tag=%f, stop=%f, word=%f, reg=%f',
        lambda_tag, lambda_stop, lambda_word, lambda_reg,
    )

    def step_fn(unused_engine, batch):
        # Move inputs to device
        images = batch.images.to(device) # shape: bs, 3, 224, 224
        reports = batch.reports.to(device) # shape: bs, n_sentences, n_words
        gt_stops = batch.stops.to(device) # shape: bs, n_sentences
        labels = batch.labels.to(device).float() # shape: bs, n_labels

        # Use labels as a distribution
        # FIXME: how do they make this work?? MSELoss() ??
        # labels = labels / (labels.sum(dim=1).unsqueeze(-1) + 1e-5)

        # Enable training
        model.train(training)
        torch.set_grad_enabled(training)

        # zero the parameter gradients
        if training:
            optimizer.zero_grad()

        # Forward pass
        out_words, out_stops, out_scores_visual, out_scores_tags, _, out_tags = model(
            images, reports=reports, free=free, max_words=max_words, max_sentences=max_sentences,
        )
        # shapes:
        # out_words: bs, n_sentences, n_words, vocab_size
        # out_stops: bs, n_sentences
        # out_scores_visual: bs, n_sentences, n_pixels
        # out_scores_tags: bs, n_sentences, n_top_tags
        # out_tags: bs, n_labels

        if not free:
            # Calculate word loss
            word_loss = word_loss_fn(out_words.permute(0, 3, 1, 2), reports)

            # Calculate stop loss
            stop_loss = stop_loss_fn(out_stops, gt_stops)

            # Tags loss
            tag_loss = tag_loss_fn(out_tags, labels)

            # Regularization
            reg_loss = _calculate_regularization(out_scores_visual, out_scores_tags)

            # Total loss
            batch_loss = lambda_tag * tag_loss \
                        + lambda_stop * stop_loss \
                        + lambda_word * word_loss \
                        + lambda_reg * reg_loss
        else:
            batch_loss = None
            tag_loss = None
            stop_loss = None
            word_loss = None
            reg_loss = None

        if training:
            batch_loss.backward()

            optimizer.step()

        out_words = out_words.detach()
        out_stops = out_stops.detach()
        # out_tags = out_tags.detach()

        return {
            'loss': batch_loss.detach() if batch_loss is not None else None,
            'word_loss': word_loss.detach() if word_loss is not None else None,
            'stop_loss': stop_loss.detach() if stop_loss is not None else None,
            'tag_loss': tag_loss.detach() if tag_loss is not None else None,
            'reg_loss': reg_loss.detach() if reg_loss is not None else None,
            'gt_stops': gt_stops,
            'flat_clean_reports_gen': _flatten_gen_reports(out_words, out_stops),
            'flat_clean_reports_gt': _flatten_gt_reports(reports),
        }
    return step_fn
