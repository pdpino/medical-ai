import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import numpy as np

from medai.datasets.common import BatchItems
from medai.utils.nlp import END_IDX, END_OF_SENTENCE_IDX, split_sentences_and_pad, PAD_IDX


def create_hierarchical_dataloader(dataset, **kwargs):
    """Creates a dataloader from a images-report dataset, considering hierarchical sequences (sentence-words).

    Outputted reports have shape (batch_size, n_sentences, n_words)
    """
    def _collate_fn(batch_tuples):
        images = []
        reports = []
        filenames = []
        max_sentence_len = -1

        # Grab images and reports
        for tup in batch_tuples:
            images.append(tup.image)
            filenames.append(tup.filename)

            report = tup.report
            report = split_sentences_and_pad(report)
            max_sentence_len = max(max_sentence_len, report.size()[-1])

            reports.append(report)

        # Pad reports to the max_sentence_len across all reports
        padded_reports = [
            pad(report, (0, max_sentence_len - report.size()[-1]))
            if report.size()[-1] < max_sentence_len else report
            for report in reports
        ]

        images = torch.stack(images)
        # shape: batch_size, channels, height, width

        reports = pad_sequence(padded_reports, batch_first=True)
        # shape: batch_size, n_sentences, n_words

        # Compute stops
        stops = [torch.zeros(report.size()[0]) for report in padded_reports]
        stops = pad_sequence(stops, batch_first=True, padding_value=1)
        # shape: batch_size, n_sentences

        return BatchItems(
            images=images,
            reports=reports,
            stops=stops,
            filenames=filenames,
        )

    return DataLoader(dataset, collate_fn=_collate_fn, **kwargs)


def _flatten_gen_reports(generated_words, stops_prediction, threshold=0.5):
    """Flattens generated reports.
    
    generated_words -- tensor of shape (batch_size, max_n_sentences, max_n_words, vocab_size)
    stops_prediction -- tensor of shape (batch_size, max_n_sentences)

    Returns:
        padded tensor of shape batch_size, n_words
    """
    # FIXME: very inefficient: for loops, and calling pad_sequence()
    texts = []

    _, n_sentences, n_words, _ = generated_words.size()
    target_len = n_sentences * n_words

    for report, stops in zip(generated_words, stops_prediction):
        text = []

        for sentence, stop in zip(report, stops):
            if stop.item() > threshold:
                break

            # REVIEW: can this be done once outside the loops?
            _, sentence = sentence.max(dim=-1)

            for word in sentence:
                word = word.item()
                text.append(word)

                if word == END_OF_SENTENCE_IDX:
                    break

        missing_words = target_len - len(text)
        if missing_words > 0:
            text += [PAD_IDX] * missing_words
        text = torch.tensor(text) # shape: n_words

        texts.append(text) # shape: batch_size, n_words

    return pad_sequence(texts, batch_first=True)


def _flatten_gt_reports(reports):
    texts = []

    batch_size = reports.size()[0]
    target_len = reports.numel() // batch_size

    for report in reports:
        text = []
        for sentence in report:
            sentence = np.trim_zeros(sentence.detach().cpu().numpy())
            if len(sentence) > 0:
                text.extend(sentence)

        missing_words = target_len - len(text)
        if missing_words > 0:
            text += [PAD_IDX] * missing_words

        texts.append(torch.tensor(text))

    return pad_sequence(texts, batch_first=True)


def get_step_fn_hierarchical(model, optimizer=None, training=True, free=False,
                             device='cuda', max_words=50, max_sentences=20):
    """Creates a step function for an Engine, considering a hierarchical dataloader."""
    word_loss_fn = nn.CrossEntropyLoss()
    stop_loss_fn = nn.BCELoss()

    assert not (free and training), 'Cant set training=True and free=True'

    def step_fn(engine, data_batch):
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
        generated_words = output_tuple[0] # shape: batch_size, max_n_sentences, max_n_words, vocab_size
        stop_prediction = output_tuple[1] # shape: batch_size, max_n_sentences

        if not free:
            # If free, outputs will have different sizes
            # TODO: pad output arrays to be able to calculate loss anyway

            # Calculate word loss
            vocab_size = generated_words.size()[-1]
            word_loss = word_loss_fn(generated_words.view(-1, vocab_size), reports.view(-1))
            
            # Calculate stop loss
            stop_loss = stop_loss_fn(stop_prediction, stop_ground_truth)
            
            # Calculate full loss
            total_loss = word_loss + stop_loss
            batch_loss = total_loss.item()
            word_loss = word_loss.item()
            stop_loss = stop_loss.item()

        else:
            batch_loss = -1
            word_loss = -1
            stop_loss = -1

        if training:
            total_loss.backward()
            optimizer.step()

        flat_reports_gen = _flatten_gen_reports(generated_words, stop_prediction)
        flat_reports = _flatten_gt_reports(reports)

        # print('AAA: ', generated_words.size())
        # print('BBB: ', reports.size())
        # print('CCC: ', flat_reports_gen.size())
        # print('DDD: ', flat_reports.size())

        return {
            'loss': batch_loss,
            'word_loss': word_loss,
            'stop_loss': stop_loss,
            'words_scores': generated_words,
            # 'reports': reports,
            'flat_reports_gen': flat_reports_gen, # shape: batch_size, n_words
            'flat_reports': flat_reports, # shape: batch_size, n_words
        }

    return step_fn