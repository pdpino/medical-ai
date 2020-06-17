import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import json

from mrg.utils import (
    PAD_TOKEN,
    PAD_IDX,
    END_TOKEN,
    END_IDX,
    START_TOKEN,
    START_IDX,
    UNKNOWN_TOKEN,
    UNKNOWN_IDX,
)

DATASET_DIR = os.environ['DATASET_DIR_IU_XRAY']


# TODO: move this to utils
def compute_vocab(reports_iterator):
    word_to_idx = {
        PAD_TOKEN: PAD_IDX,
        START_TOKEN: START_IDX,
        END_TOKEN: END_IDX,
        UNKNOWN_TOKEN: UNKNOWN_IDX,
    }

    for report in reports_iterator:
        for token in report:
            if token not in word_to_idx:
                word_to_idx[token] = len(word_to_idx)

    return word_to_idx


def _reports_iterator(reports):
    for report in reports:
        yield report['clean_text'].split()


def _get_default_image_transformation(image_size=512):
    return transforms.Compose([transforms.Resize((image_size, image_size)),
                               transforms.ToTensor(),
                              ])

class IUXRayDataset(Dataset):
    def __init__(self, dataset_type='train', max_samples=None, sort_samples=True,
                 vocab=None):
        """Create a Dataset object."""
        if dataset_type not in ['train', 'val', 'test']:
            raise ValueError('No such type, must be train, val, or test')
        
        self.dataset_type = dataset_type # TODO: use dataset type?
        self.image_format = 'RGB'
        self.transform = _get_default_image_transformation()
        
        self.images_dir = os.path.join(DATASET_DIR, 'images')
        self.reports_dir = os.path.join(DATASET_DIR, 'reports')

        # Load reports
        reports_fname = os.path.join(self.reports_dir, 'reports.clean.json')
        with open(reports_fname, 'r') as f:
            reports = list(json.load(f).values())

        # Filter by train, val, test
        list_fname = os.path.join(self.reports_dir, f'{dataset_type}.txt')
        with open(list_fname, 'r') as f:
            reports_from_split = set(l.strip() for l in f.readlines())
        reports = [rep for rep in reports if rep['filename'] in reports_from_split]

        # Keep only max images
        if max_samples is not None:
            reports = reports[:max_samples]

        # Save amounts
        self.n_reports = len(reports)
        self.n_images = sum(len(report['images']) for report in reports)

        self._preprocess_reports(reports, sort_samples=sort_samples, vocab=vocab)
        
    def size(self):
        return (self.n_images, self.n_reports)

    def get_by_name(self, report_name):
        # TODO
        raise NotImplementedError
    
    def __len__(self):
        return self.n_reports

    def __getitem__(self, idx):
        report = self.reports[idx]

        image = self.load_image(report['image_name'])

        return image, report['tokens_idxs']

    def load_image(self, image_name):
        image_path = os.path.join(self.images_dir, f'{image_name}.png')
        try:
            image = Image.open(image_path).convert(self.image_format)
        except OSError as e:
            print(f'({self.dataset_type}) Failed to load image, may be broken: {image_path}')
            print(e)
            raise

        image = self.transform(image)
        return image

    def get_vocab(self):
        return self.word_to_idx

    def _preprocess_reports(self, reports, sort_samples=True, vocab=None):
        # Save a name_to_idx dict
        self.name_to_idx = {}

        for index, report in enumerate(reports):
            name = report['filename']

            self.name_to_idx[name] = index

        if vocab is None:
            # Compute word_to_idx dictionary
            self.word_to_idx = compute_vocab(_reports_iterator(reports))
        else:
            self.word_to_idx = vocab

        # Compute final reports array
        self.reports = []
        for report in reports:
            tokens = report['clean_text'].split()

            tokens_idxs = [
                self.word_to_idx.get(token, UNKNOWN_IDX)
                for token in tokens
            ]

            for image in report['images']:
                self.reports.append({
                    'tokens_idxs': torch.tensor(tokens_idxs + [END_IDX]),
                    'image_name': image['id'],
                })

        if sort_samples:
            self.reports = sorted(self.reports, key=lambda x:len(x['tokens_idxs']))
