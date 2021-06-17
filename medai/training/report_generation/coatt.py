"""Custom step_fn for the CoAtt model."""
import logging
import torch
from torch import nn

from medai.training.report_generation.hierarchical import _flatten_gt_reports, _flatten_gen_reports

LOGGER = logging.getLogger(__name__)

def get_step_fn_coatt(model, optimizer=None, training=True,
                      lambda_tag=1, lambda_stop=1, lambda_word=1,
                      clip=0,
                      device='cuda',
                      **unused_kwargs,
                      ):
    """Return a step-fn for the CoAtt model.

    Notice the stops are inverted inside the co-att model (with respect to default medai behavior):
        1 means continue, 0 means stop.
    """
    # mse_criterion = nn.MSELoss(reduction='mean')
    tag_criterion = nn.BCEWithLogitsLoss()

    LOGGER.info(
        'Using lambdas: tag=%f, stop=%f, word=%f',
        lambda_tag, lambda_stop, lambda_word,
    )

    def step_fn(unused_engine, batch):
        # Move inputs to device
        images = batch.images.to(device) # shape: bs, 3, 224, 224
        labels = batch.labels.to(device) # shape: bs, n_labels
        reports = batch.reports.to(device) # shape: bs, n_sentences, n_words
        gt_stops = (1 - batch.stops.to(device)).long() # shape: bs, n_sentences
        # stops are inverted!

        # Enable training
        model.train(training)
        torch.set_grad_enabled(training)

        # zero the parameter gradients
        if training:
            optimizer.zero_grad()

        # Forward pass
        out_words, out_stops, tags, batch_stop_loss, batch_word_loss = model(
            images, reports, gt_stops,
        )
        # shapes:
        # words: bs, n_sentences, n_words, vocab_size
        # out_stops: bs, n_sentences
        # tags: bs, n_labels
        # losses: 1

        out_stops = 1 - out_stops
        # stops are inverted!!

        # Tags loss
        batch_tag_loss = tag_criterion(tags, labels.float())

        # Total loss
        batch_loss = lambda_tag * batch_tag_loss \
                     + lambda_stop * batch_stop_loss \
                     + lambda_word * batch_word_loss

        if training:
            batch_loss.backward()

            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.sentence_model.parameters(), clip)
                torch.nn.utils.clip_grad_norm_(model.word_model.parameters(), clip)

            optimizer.step()

        out_words = out_words.detach()
        out_stops = out_stops.detach()

        return {
            'loss': batch_loss.detach(),
            'word_loss': batch_word_loss.detach(),
            'stop_loss': batch_stop_loss.detach(),
            'tag_loss': batch_tag_loss.detach(),
            'gt_stops': gt_stops,
            'flat_clean_reports_gen': _flatten_gen_reports(out_words, out_stops),
            'flat_clean_reports_gt': _flatten_gt_reports(reports),
        }
    return step_fn
