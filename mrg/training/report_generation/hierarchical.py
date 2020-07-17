import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import numpy as np

from mrg.utils.nlp import END_IDX, END_OF_SENTENCE_IDX

def _split_sentences_and_pad(report, end_of_sentence_idx=END_OF_SENTENCE_IDX):
    """Splits a report into sentences and pads them.
    
    Args:
        report -- tensor or list of shape (n_words)
        end_of_sentence_idx -- int indicating idx of the end-of-sentence token
    Returns:
        report (tensor) of shape (n_sentences, n_words)
    """
    if not isinstance(report, torch.Tensor):
        report = torch.tensor(report)

    # Index positions of end-of-sentence tokens
    end_positions = (report == end_of_sentence_idx).nonzero().view(-1)

    # Transform it to count of items
    end_counts = end_positions + 1
    
    # Calculate sentence sizes, by subtracting index positions to the one before
    shifted_counts = torch.cat((torch.zeros(1).long(), end_counts), dim=0)[:-1]
    split_sizes = (end_counts - shifted_counts).tolist()
    
    # Split into sentences
    try:
        sentences = torch.split(report, split_sizes)
    except:
        # DEBUG: catch case where this fails
        print('FAILED HERE:')
        print(report)
        print(split_sizes)
        print(end_positions)
        print(end_counts)
        # return dummy result
        return pad_sequence([torch.tensor(3), torch.tensor(4)], batch_first=True)
    
    return pad_sequence(sentences, batch_first=True)


def create_hierarchical_dataloader(dataset, **kwargs):
    """Creates a dataloader from a images-report dataset, considering hierarchical sequences (sentence-words).

    Outputted reports have shape (batch_size, n_sentences, n_words)
    """
    def _collate_fn(batch_tuples):
        images = []
        reports = []
        max_sentence_len = -1

        # Grab images and reports
        for image, seq_out in batch_tuples:
            images.append(image)

            # Last sentence must end with a dot
            if seq_out[-1] != END_OF_SENTENCE_IDX:
                seq_out.append(END_OF_SENTENCE_IDX)

            report = _split_sentences_and_pad(seq_out)
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

        return images, reports, stops

    return DataLoader(dataset, collate_fn=_collate_fn, **kwargs)


def _flatten_gen_reports(reports_gen, stops_gen, threshold=0.5):
    # FIXME: very inefficient: for loops, and calling pad_sequence()
    texts = []

    for report, stops in zip(reports_gen, stops_gen):
        text = []

        for sentence, stop in zip(report, stops):
            if stop.item() > threshold:
                break

            _, sentence = sentence.max(dim=-1)

            for word in sentence:
                if word == END_OF_SENTENCE_IDX:
                    break

                text.append(word.item())
        texts.append(torch.tensor(text))

    return pad_sequence(texts, batch_first=True)


def _flatten_gt_reports(reports):
    texts = []

    for report in reports:
        text = []
        for sentence in report:
            sentence = np.trim_zeros(sentence.detach().cpu().numpy())
            if len(sentence) > 0:
                text.extend(sentence)

        texts.append(torch.tensor(text))

    return pad_sequence(texts, batch_first=True)


def get_step_fn_hierarchical(model, optimizer=None, training=True, device='cuda'):
    """Creates a step function for an Engine, considering a hierarchical dataloader."""
    word_loss_fn = nn.CrossEntropyLoss()
    stop_loss_fn = nn.BCELoss()

    def step_fn(engine, data_batch):
        # Images
        images = data_batch[0].to(device)
        # shape: batch_size, 3, height, width

        # Reports (hierarchical), as word ids
        reports = data_batch[1].to(device).long()
        # shape: batch_size, max_n_sentences, max_n_words
        # _, max_sentence_len, max_n_words = reports.size()
        
        # Stop ground-truth
        stop_ground_truth = data_batch[2].to(device)
        # shape: batch_size, n_sentences
        
        # Enable training
        model.train(training)
        torch.set_grad_enabled(training) # enable recording gradients

        # zero the parameter gradients
        if training:
            optimizer.zero_grad()

        # Pass thru the model
        output_tuple = model(images, reports)

        # Calculate word loss
        generated_words = output_tuple[0]
        # shape: batch_size, max_n_sentences, max_n_words, vocab_size

        vocab_size = generated_words.size()[-1]
        word_loss = word_loss_fn(generated_words.view(-1, vocab_size), reports.view(-1))
        
        # Calculate stop loss
        stop_prediction = output_tuple[1]
        # shape: batch_size, max_n_sentences

        stop_loss = stop_loss_fn(stop_prediction, stop_ground_truth)
        
        # Calculate full loss
        total_loss = word_loss + stop_loss
        batch_loss = total_loss.item()

        if training:
            total_loss.backward()
            optimizer.step()

        # return batch_loss, generated_words, reports, stop_prediction, stop_ground_truth
        flattened_reports_gen = _flatten_gen_reports(generated_words, stop_prediction)
        flattened_reports_gt = _flatten_gt_reports(reports)
        return (
            batch_loss,
            generated_words,
            reports,
            flattened_reports_gen,
            flattened_reports_gt,
        )

    return step_fn