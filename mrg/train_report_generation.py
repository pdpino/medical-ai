import time
import argparse

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from ignite.engine import Engine, Events
from ignite.handlers import Timer, Checkpoint, DiskSaver

from mrg.datasets.iu_xray import IUXRayDataset
from mrg.metrics.report_generation import attach_metrics_report_generation
from mrg.models.report_generation.cnn_to_seq import CNN2Seq
from mrg.models.report_generation.decoder_lstm import LSTMDecoder
from mrg.models.classification import (
    AVAILABLE_CLASSIFICATION_MODELS,
    init_empty_model,
)
from mrg.models.checkpoint import (
    CompiledModel,
    attach_checkpoint_saver,
    load_compiled_model_classification,
)
from mrg.tensorboard import TBWriter
from mrg.utils import get_timestamp, duration_to_str


def create_pad_dataloader(dataset, batch_size=128, shuffle=False, **kwargs):
    """Creates a dataloader from a images-report dataset.
    
    Pads the output sequence.
    """
    def collate_fn(batch_tuples):
        images = []
        batch_seq_out = []
        for image, seq_out in batch_tuples:
            images.append(image)
            batch_seq_out.append(seq_out)

        images = torch.stack(images)
        batch_seq_out = pad_sequence(batch_seq_out, batch_first=True)
        return images, batch_seq_out

    dataloader = DataLoader(dataset, batch_size, collate_fn=collate_fn,
                            shuffle=shuffle, **kwargs)
    return dataloader


def get_step_fn(model, optimizer=None, training=True, device='cuda'):
    """Creates a step function for an Engine."""
    loss_fn = nn.CrossEntropyLoss()

    def step_fn(engine, data_batch):
        # Images
        images = data_batch[0].to(device)
        # shape: batch_size, 3, height, width

        # Reports, as word ids
        reports = data_batch[1].to(device).long() # shape: batch_size, max_sentence_len
        _, max_sentence_len = reports.size()
        
        # Enable training
        model.train(training)
        torch.set_grad_enabled(training) # enable recording gradients

        # zero the parameter gradients
        if training:
            optimizer.zero_grad()

        # Pass thru the model
        output_tuple = model(images, max_sentence_len, reports)

        generated_words = output_tuple[0]
        _, _, vocab_size = generated_words.size()
        # shape: batch_size, n_sentences, vocab_size

        # Compute classification loss
        loss = loss_fn(generated_words.view(-1, vocab_size), reports.view(-1))
        
        batch_loss = loss.item()

        if training:
            loss.backward()
            optimizer.step()

        return batch_loss, generated_words, reports

    return step_fn


def train_model(run_name,
                compiled_model,
                train_dataloader,
                val_dataloader,
                n_epochs=1,
                debug=True,
                print_metrics=['loss', 'word_acc'],
                device='cuda',
               ):
    # Prepare run stuff
    tb_writer = TBWriter(run_name, classification=False, debug=debug)
    initial_epoch = compiled_model.get_current_epoch()

    # Unwrap stuff
    model, optimizer = compiled_model.get_model_optimizer()

    # Create validator engine
    validator = Engine(get_step_fn(model, training=False, device=device))
    attach_metrics_report_generation(validator)
    
    # Create trainer engine
    trainer = Engine(get_step_fn(model, optimizer=optimizer, training=True, device=device))
    attach_metrics_report_generation(trainer)
    
    # Create Timer to measure wall time between epochs
    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, step=Events.EPOCH_COMPLETED)

    # Attach checkpoint
    attach_checkpoint_saver(run_name,
                            compiled_model,
                            trainer,
                            classification=False,
                            debug=debug,
                           )

    @trainer.on(Events.EPOCH_COMPLETED)
    def tb_write_metrics(trainer):
        # Run on evaluation
        validator.run(val_dataloader, 1)

        # State
        epoch = trainer.state.epoch + initial_epoch
        max_epochs = trainer.state.max_epochs + initial_epoch
        train_metrics = trainer.state.metrics
        val_metrics = validator.state.metrics
        
        # Save state
        compiled_model.save_current_epoch(epoch)
        # run_state.save_state(epoch)

        # Common time
        wall_time = time.time()
        
        # Log to TB
        tb_writer.write_histogram(model, epoch, wall_time)
        tb_writer.write_metrics(train_metrics, 'train', epoch, wall_time)
        tb_writer.write_metrics(val_metrics, 'val', epoch, wall_time)
        
        # Print metrics
        print_str = f'Finished epoch {epoch}/{max_epochs}'
        for metric in print_metrics:
            train_value = train_metrics.get(metric, -1)
            val_value = val_metrics.get(metric, -1)
            metric_str = f' {metric} {train_value:.4f} {val_value:.4f},'
            print_str += metric_str

        print_str += f' {duration_to_str(timer._elapsed())}'
        print(print_str)

    # Train!
    print('-' * 50)
    print('Training...')
    trainer.run(train_dataloader, n_epochs)

    # Capture time
    secs_per_epoch = timer.value()
    duration_per_epoch = duration_to_str(secs_per_epoch)
    print('Average time per epoch: ', duration_per_epoch)
    print('-'*50)
    
    # Close stuff
    tb_writer.close()

    return


def main(run_name,
         batch_size=15,
         teacher_forcing=True,
         embedding_size=100,
         hidden_size=100,
         lr=0.0001,
         n_epochs=10,
         cnn_run_name=None,
         cnn_model_name='resnet-50',
         cnn_imagenet=True,
         cnn_freeze=False,
         max_samples=None,
         debug=True,
         multiple_gpu=False,
         device='cuda',
         ):

    # Create run name
    run_name = f'{run_name}_lr{lr}'

    print('Run: ', run_name)

    # Load data
    train_dataset = IUXRayDataset(dataset_type='train', max_samples=max_samples)
    val_dataset = IUXRayDataset(dataset_type='val',
                                vocab=train_dataset.get_vocab(),
                                max_samples=max_samples,
                                )

    train_dataloader = create_pad_dataloader(train_dataset, batch_size=batch_size)
    val_dataloader = create_pad_dataloader(val_dataset, batch_size=batch_size)


    # Create CNN
    if cnn_run_name:
        # Load pretrained
        compiled_cnn = load_compiled_model_classification(cnn_run_name,
                                                          debug=debug,
                                                          device=device)
        cnn = compiled_cnn.model
    else:
        # Create new
        cnn = init_empty_model(cnn_model_name,
                               labels=[],
                               imagenet=cnn_imagenet,
                               freeze=cnn_freeze,
                               ).to(device)

    # Create decoder
    decoder = LSTMDecoder(len(train_dataset.word_to_idx),
                          embedding_size,
                          hidden_size,
                          teacher_forcing=teacher_forcing,
                          ).to(device)

    # Full model
    model = CNN2Seq(cnn, decoder).to(device)

    if multiple_gpu:
        # TODO: use DistributedDataParallel instead
        model = nn.DataParallel(model)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Compiled model
    compiled_model = CompiledModel(model, optimizer)


    # Train
    train_model(run_name, compiled_model, train_dataloader, val_dataloader,
                n_epochs=n_epochs, device=device, debug=debug)


    print(f'Finished run: {run_name}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to load (debugging)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-6,
                        help='Learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('-emb', '--embedding-size', type=int, default=100,
                        help='Embedding size of the decoder')
    parser.add_argument('-hs', '--hidden-size', type=int, default=100,
                        help='Hidden size of the decoder')
    parser.add_argument('-notf', '--no-teacher-forcing', action='store_true',
                        help='If present, does not use teacher forcing')
    parser.add_argument('--multiple-gpu', action='store_true',
                        help='Use multiple gpus')
    parser.add_argument('--no-debug', action='store_true',
                        help='If is a non-debugging run')

    cnn_group = parser.add_argument_group('CNN')
    cnn_group.add_argument('-c', '--cnn', type=str, default=None,
                        choices=AVAILABLE_CLASSIFICATION_MODELS,
                        help='Choose base CNN class (create new)')
    cnn_group.add_argument('-noig', '--no-imagenet', action='store_true',
                        help='If present, dont use imagenet pretrained weights')
    cnn_group.add_argument('-frz', '--freeze', action='store_true',
                        help='If present, freeze base cnn parameters (only train FC layers)')
    cnn_group.add_argument('-cp', '--cnn-pretrained', type=str, default=None,
                        help='Run name of a pretrained CNN')

    args = parser.parse_args()

    if args.cnn is None and args.cnn_pretrained is None:
        raise Exception('Must choose one of cnn or cnn_pretrained')

    return args


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args()

    # TODO: once model resuming is implemented, enable names other than timestamp
    run_name = get_timestamp()

    start_time = time.time()

    main(run_name,
         batch_size=args.batch_size,
         teacher_forcing=not args.no_teacher_forcing,
         embedding_size=args.embedding_size,
         hidden_size=args.hidden_size,
         lr=args.learning_rate,
         n_epochs=args.epochs,
         cnn_run_name=args.cnn_pretrained,
         cnn_model_name=args.cnn,
         cnn_imagenet=not args.no_imagenet,
         cnn_freeze=args.freeze,
         max_samples=args.max_samples,
         debug=not args.no_debug,
         multiple_gpu=args.multiple_gpu,
         device=device,
         )

    total_time = time.time() - start_time
    print(f'Total time: {duration_to_str(total_time)}')
