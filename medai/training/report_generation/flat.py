import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from medai.utils.nlp import END_IDX


def create_flat_dataloader(dataset, **kwargs):
    """Creates a dataloader from a images-report dataset, considering flat word sequences.
    
    Outputed reports have shape (batch_size, n_words)
    Adds END_TOKEN to the end of the sentences, and pads the output sequence.
    """
    def _collate_fn(batch_tuples):
        images = []
        batch_seq_out = []
        for image, seq_out in batch_tuples:
            images.append(image)
            batch_seq_out.append(torch.tensor(seq_out + [END_IDX]))

        images = torch.stack(images)
        batch_seq_out = pad_sequence(batch_seq_out, batch_first=True)
        return images, batch_seq_out

    return DataLoader(dataset, collate_fn=_collate_fn, **kwargs)


def get_step_fn_flat(model, optimizer=None, training=True, device='cuda'):
    """Creates a step function for an Engine."""
    loss_fn = nn.CrossEntropyLoss()

    def step_fn(engine, data_batch):
        # Images
        images = data_batch[0].to(device)
        # shape: batch_size, 3, height, width

        # Reports, as word ids
        reports = data_batch[1].to(device).long()
        # shape: batch_size, max_sentence_len
        
        # Enable training
        model.train(training)
        torch.set_grad_enabled(training) # enable recording gradients

        # zero the parameter gradients
        if training:
            optimizer.zero_grad()

        # Pass thru the model
        output_tuple = model(images, reports)

        generated_words = output_tuple[0]
        _, _, vocab_size = generated_words.size()
        # shape: batch_size, n_sentences, vocab_size

        # Compute word classification loss
        loss = loss_fn(generated_words.view(-1, vocab_size), reports.view(-1))
        
        batch_loss = loss.item()

        if training:
            loss.backward()
            optimizer.step()

        return batch_loss, generated_words, reports

    return step_fn