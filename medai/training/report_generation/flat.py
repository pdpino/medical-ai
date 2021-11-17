import torch
from torch import nn

from medai.utils.nlp import PAD_IDX, END_IDX, START_IDX


def _clean_gen_reports(gen_words):
    """Cleans the generated reports.

    Args:
        gen_words -- tensor of shape batch_size, n_words[, vocab_size]
    Returns:
        list of lists with reports, shape: batch_size, n_words_per_report
        (with token indexes, not words).
    """
    if isinstance(gen_words, torch.Tensor):
        if gen_words.ndim == 3:
            gen_words = gen_words.argmax(-1)
            # shape: bs, n_words
        else:
            assert gen_words.ndim == 2
            # assume shape: bs, n_words
    else:
        assert isinstance(gen_words, list)

    clean_reports = []
    for report in gen_words:
        # report shape: n_words

        # Remove PAD_IDX
        report = report[(report != PAD_IDX) & (report != START_IDX)]
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


def clean_gt_reports(gt_reports):
    """Cleans the GT reports.

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
                     device='cuda', max_words=200, temperature=1,
                     lambda_att=0, beam_size=0,
                     **unused_kwargs):
    """Creates a step function for an Engine."""
    if beam_size > 0:
        assert free, 'Cannot set beam_size>0 and free=False'
        assert not training, 'Cannot set beam_size>0 and training=True'
        assert isinstance(model.cnn, nn.Module)
        assert isinstance(model.decoder, nn.Module)

        def step_fn_beam_size(unused_engine, data_batch):
            """Step function using beam search.

            Decoder must accept beam_size, and return tensor of shape = (n_words,)
            Return specific function, since is slightly different
            """
            # Images
            images = data_batch.images.to(device) # batch_size, 3, height, width

            model.eval()

            with torch.no_grad():
                features = model.cnn.features(images)
                # shape: batch_size, n_features, f-height, f-width

                generated_words = [] # shape: batch_size, n_words (list of tensors)

                for feature in features:
                    output_tuple = model.decoder(
                        feature.unsqueeze(0), # 1, n_features, f-height, f-width
                        reports=None, free=True,
                        max_words=max_words,
                        beam_size=beam_size,
                    )

                    out_words = output_tuple[0] # shape: n_words
                    assert out_words.ndim == 1
                    generated_words.append(out_words)

            # Reports, as word ids
            # Not necessary to send to GPU, since are cleaned as lists right away
            reports = data_batch.reports # batch_size, max_sentence_len

            return {
                'loss': None,
                'att_loss': None,
                'word_loss': None,
                'flat_clean_reports_gen': _clean_gen_reports(generated_words),
                'flat_clean_reports_gt': clean_gt_reports(reports),
            }

        return step_fn_beam_size

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
            loss_word = loss_fn(generated_words.permute(0, 2, 1) / temperature, reports)

            if lambda_att > 0:
                # Compute attention loss (show-att-tell)
                assert len(output_tuple) > 1, 'output_tuple only has 1 elem and lambda_att > 0'
                scores_out = output_tuple[1] # shape: bs, n_words, height, width
                assert scores_out is not None, 'scores_out is None and lambda_att > 0'

                loss_att = torch.mean((1 - scores_out.sum(dim=1))**2)
                loss = loss_word + lambda_att * loss_att
            else:
                loss = loss_word
                loss_att = None
        else:
            loss = None
            loss_att = None
            loss_word = None

        if training:
            loss.backward()
            optimizer.step()

        generated_words = generated_words.detach()

        return {
            'loss': loss.detach() if loss is not None else None,
            'att_loss': loss_att.detach() if loss_att is not None else None,
            'word_loss': loss_word.detach() if loss_word is not None else None,
            'flat_clean_reports_gen': _clean_gen_reports(generated_words),
            'flat_clean_reports_gt': clean_gt_reports(reports),
        }

    return step_fn
