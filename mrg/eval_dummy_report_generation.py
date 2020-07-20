import argparse
import torch

from mrg.datasets.iu_xray import IUXRayDataset
from mrg.models.report_generation.dummy.constant import ConstantReport
from mrg.models.report_generation.dummy.common_words import MostCommonWords
from mrg.models.report_generation.dummy.common_sentences import MostCommonSentences
from mrg.training.report_generation.flat import (
    create_flat_dataloader,
    get_step_fn_flat,
)
from mrg.training.report_generation.hierarchical import (
    create_hierarchical_dataloader,
    get_step_fn_hierarchical,
)
from mrg.train_report_generation import evaluate_and_save
from mrg.utils import get_timestamp


_AVAILABLE_DUMMY_MODELS = [
    'constant',
    'common-words',
    'common-sentences',
]


DUMMY_REPORT = '''the heart is normal in size . the mediastinum is unremarkable . 
the lungs are clear .
there is no pneumothorax or pleural effusion . no focal airspace disease .
no pleural effusion or pneumothorax . END'''

def evaluate_dummy_model(model_name,
                         batch_size=20,
                         k_first=100,
                         debug=True,
                         device='cuda',
                         ):
    # Run name
    run_name = f'{get_timestamp()}_dummy-{model_name}'
    if 'common-' in model_name:
        run_name += f'-{str(k_first)}'

    # Load datasets
    train_dataset = IUXRayDataset('train')
    vocab = train_dataset.get_vocab()
    val_dataset = IUXRayDataset('val', vocab=vocab)
    test_dataset = IUXRayDataset('test', vocab=vocab)    


    # Choose model
    if model_name == 'constant':
        model = ConstantReport(vocab, DUMMY_REPORT)

    elif model_name == 'common-words':
        model = MostCommonWords(train_dataset, k_first)

    elif model_name == 'common-sentences':
        model = MostCommonSentences(train_dataset, k_first)
    
    else:
        raise Exception(f'Model not recognized: {model_name}')



    # Decide hierarchical
    if model.hierarchical:
        get_step_fn = get_step_fn_hierarchical
        create_dataloader = create_hierarchical_dataloader
    else:
        get_step_fn = get_step_fn_flat
        create_dataloader = create_flat_dataloader


    # Create dataloaders
    train_dataloader = create_dataloader(train_dataset, batch_size=batch_size)
    val_dataloader = create_dataloader(val_dataset, batch_size=batch_size)
    test_dataloader = create_dataloader(test_dataset, batch_size=batch_size)

    dataloaders = [
        train_dataloader,
        val_dataloader,
        test_dataloader
    ]

    # Evaluate
    evaluate_and_save(run_name,
                      model,
                      dataloaders,
                      hierarchical=model.hierarchical,
                      debug=debug,
                      device=device,
                      )

    print('Evaluated ', run_name)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', type=str, default=None,
                        choices=_AVAILABLE_DUMMY_MODELS, help='Dummy model to use')
    parser.add_argument('-bs', '--batch_size', type=int, default=20,
                        help='Batch size to use')
    parser.add_argument('-k', '--k-first', type=int, default=100,
                        help='Top k selected for common-words and common-sentences')
    parser.add_argument('--no-debug', action='store_true',
                        help='If is a non-debugging run')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to load (debugging)')
    parser.add_argument('--cpu', action='store_true',
                    help='Use CPU only')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')

    evaluate_dummy_model(args.model_name,
                         batch_size=args.batch_size,
                         k_first=args.k_first,
                         debug=not args.no_debug,
                         device=device,
                         )


