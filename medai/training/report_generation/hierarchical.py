import torch
from torch import nn

from medai.utils.nlp import PAD_IDX, END_OF_SENTENCE_IDX, END_IDX
from medai.losses.out_of_target import OutOfTargetSumLoss



def _flatten_gen_reports(generated_words, stops_prediction, threshold=0.5):
    """Flattens generated reports.

    Args:
        generated_words -- tensor of shape
            (batch_size, n_max_sentences, n_max_words_per_sentence, vocab_size)
        stops_prediction -- tensor of shape (batch_size, n_max_sentences)

    Returns:
        list of lists of shape batch_size, n_words_per_report
    """
    texts = []

    generated_words = generated_words.argmax(dim=-1)
    # shape: batch_size, n_max_sentences, n_max_words_per_sentence

    batch_size, n_max_sentences = generated_words.size()[:2]

    # Create dummy end-of-sentence token array
    # pylint: disable=not-callable
    extra_end_of_sentence = torch.tensor(
        [END_OF_SENTENCE_IDX],
        device=generated_words.device,
    ).expand(batch_size, n_max_sentences, -1)
    # shape: batch_size, n_max_sentences, 1

    # Ensure all sentences end with a dot (END_OF_SENTENCE), by concating the dummy array
    # The extra dots are eliminated anyway
    generated_words = torch.cat((generated_words, extra_end_of_sentence), dim=-1)
    # shape: batch_size, n_max_sentences, n_max_words_per_sentence + 1

    valid_sentences = torch.where(
        stops_prediction < threshold,
        torch.ones_like(stops_prediction),
        torch.zeros_like(stops_prediction),
    )
    # shape: batch_size, n_max_sentences
    # Binary indicator of valid sentences: 1 if the sentence is valid, 0 otherwise

    for report, valid in zip(generated_words, valid_sentences):
        # report shape: n_sentences, n_words+1
        # valid shape: n_sentences

        report = report[valid.nonzero(as_tuple=True)]
        # shape: n_valid_sentences, n_words+1

        dot_positions = (report == END_OF_SENTENCE_IDX).type(torch.uint8)
        # shape: n_valid_sentences, n_words+1
        # has 1s in positions with a dot, 0s elsewhere

        dot_positions = torch.cumsum(torch.cumsum(dot_positions, dim=1), dim=1)
        # shape: n_valid_sentences, n_words+1
        # will have 0s in the first sentence, 1 in the first dot, and >1 afterwards
        # Two cumsums are needed so there are no 1s right after the first dot
        # REVIEW: is there are more efficient and elegant way to do this? Avoid for loops!

        # Keep only words before a dot
        report = report[dot_positions <= 1]
        # shape: n_words_before_dot

        report = report[(report != END_IDX) & (report != PAD_IDX)]
        # shape: n_clean_words

        texts.append(report.tolist())

    return texts


def _flatten_gt_reports(reports):
    """Flats ground truth hierarchical reports.

    Args:
        reports -- tensor of shape (batch_size, n_max_sentences,
            n_max_words_per_sentence)
    Returns:
        list of lists of shape (batch_size, n_words_per_report)
    """
    texts = [
        report[(report != PAD_IDX).nonzero(as_tuple=True)].tolist()
        for report in reports
    ]

    return texts


def get_step_fn_hierarchical(model, optimizer=None, training=True,
                             free=False, device='cuda',
                             supervise_attention=False,
                             supervise_sentences=False,
                             word_lambda=1, stop_lambda=1, att_lambda=1, sentence_lambda=1,
                             max_words=50, max_sentences=20, **unused_kwargs):
    """Creates a step function for an Engine, considering a hierarchical dataloader."""
    word_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    stop_loss_fn = nn.BCELoss()
    att_loss_fn = OutOfTargetSumLoss()
    sentence_loss_fn = nn.MSELoss(reduction='none')

    assert not (free and training), 'Cannot set training=True and free=True'

    def step_fn(unused_engine, data_batch):
        # Images
        images = data_batch.images.to(device)
        # shape: batch_size, 3, height, width

        # Reports (hierarchical), as word ids
        reports = data_batch.reports.to(device).long()
        # shape: batch_size, max_n_sentences, max_n_words

        # Stop ground-truth
        stop_ground_truth = data_batch.stops.to(device)
        # shape: batch_size, n_sentences

        # Enable training
        model.train(training)
        torch.set_grad_enabled(training) # enable recording gradients

        # zero the parameter gradients
        if training:
            optimizer.zero_grad()

        # Pass thru the model
        output_tuple = model(images, reports, free=free,
                             max_words=max_words, max_sentences=max_sentences)
        generated_words = output_tuple[0] # batch_size, max_n_sentences, max_n_words, vocab_size
        stop_prediction = output_tuple[1] # batch_size, max_n_sentences

        if data_batch.masks is not None:
            gt_masks = data_batch.masks.to(device) # shape: batch_size, n_sentences, height, width
        else:
            gt_masks = None

        gen_masks = output_tuple[2] # batch_size, n_sentences, features-height, features-width

        if not free:
            # If free, outputs will have different sizes
            # TODO: pad output arrays to be able to calculate loss anyway

            # Calculate word loss
            word_loss = word_loss_fn(generated_words.permute(0, 3, 1, 2), reports)

            # Calculate stop loss
            stop_loss = stop_loss_fn(stop_prediction, stop_ground_truth)

            # Calculate full loss
            total_loss = word_lambda * word_loss + stop_lambda * stop_loss

            if supervise_attention:
                att_loss = att_loss_fn(gen_masks, gt_masks, stop_ground_truth)
                total_loss += att_lambda * att_loss
            else:
                att_loss = -1

            if supervise_sentences:
                sentence_loss = sentence_loss_fn(
                    output_tuple[3], # shape: bs, n_sentences, emb_size
                    data_batch.sentence_embeddings.to(device), # shape: bs, n_sentences, emb_size
                ) # shape: bs, n_sentences, emb_size

                # Use only GT sentences
                sentence_loss = sentence_loss[(stop_ground_truth == 0)].mean()
                # shape: 1

                total_loss += sentence_lambda * sentence_loss
            else:
                sentence_loss = -1

            batch_loss = total_loss.item()
            word_loss = word_loss.item()
            stop_loss = stop_loss.item()

        else:
            batch_loss = -1
            word_loss = -1
            stop_loss = -1
            att_loss = -1
            sentence_loss = -1

        if training:
            total_loss.backward()
            optimizer.step()

        generated_words = generated_words.detach()
        stop_prediction = stop_prediction.detach()
        if gen_masks is not None:
            gen_masks = gen_masks.detach()

        return {
            'loss': batch_loss,
            'word_loss': word_loss,
            'stop_loss': stop_loss,
            'att_loss': att_loss,
            'sentence_loss': sentence_loss,
            'flat_clean_reports_gen': _flatten_gen_reports(generated_words, stop_prediction),
            'flat_clean_reports_gt': _flatten_gt_reports(reports),
            'gt_masks': gt_masks,
            'gen_masks': gen_masks,
            'gt_stops': stop_ground_truth,
        }

    return step_fn
