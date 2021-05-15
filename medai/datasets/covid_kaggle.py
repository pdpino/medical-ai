import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from medai.datasets.common import BatchItem

DATASET_DIR = os.environ.get('DATASET_DIR_COVID_KAGGLE')

_FOLDERS = [
    'COVID-19',
    'Viral Pneumonia',
    'NORMAL',
]

def _get_default_image_transformation(image_size=(512, 512)):
    # FIXME: wrong values
    mean = [0.4872, 0.4875, 0.4876]
    sd = [0.0352, 0.0352, 0.0352]
    return transforms.Compose([transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, sd)
                              ])

class CovidKaggleDataset(Dataset):
    def __init__(self, dataset_type='train', max_samples=None, image_size=(512, 512), **kwargs):
        super().__init__()

        raise NotImplementedError('CovidKaggleDataset mean and std')

        # pylint: disable=unreachable

        if DATASET_DIR is None:
            raise Exception('DATASET_DIR_COVID_KAGGLE not found in env variables')

        if dataset_type not in ['train', 'val', 'test']:
            raise ValueError('No such type, must be train, val, or test')

        if kwargs.get('labels', None) is not None:
            print('Labels selection in CovidKaggle dataset is not implemented yet, ignoring')

        self.dataset_type = dataset_type
        self.image_format = 'RGB'
        self.image_size = image_size
        self.transform = _get_default_image_transformation(self.image_size)

        self.labels = ['covid', 'pneumonia', 'normal']
        self.multilabel = False

        # Filter by train, val, test
        list_fname = os.path.join(DATASET_DIR, f'{dataset_type}.txt')
        with open(list_fname, 'r') as f:
            images_list = set(l.strip() for l in f.readlines())

        # Load image metadata (folder/name, label)
        self.images_metadata = []

        for label_idx, label in enumerate(_FOLDERS):
            folder = os.path.join(DATASET_DIR, label)
            folder_images = [i for i in os.listdir(folder) if i in images_list]

            self.images_metadata.extend([
                (os.path.join(label, image), label_idx)
                for image in folder_images
            ])

        # Keep only max images
        if max_samples is not None:
            self.images_metadata = self.images_metadata[:max_samples]

    def size(self):
        return len(self.images_metadata), len(self.labels)

    def __len__(self):
        return len(self.images_metadata)

    def __getitem__(self, idx):
        image_name, label = self.images_metadata[idx]
        # image_name includes label-folder and extension

        image_path = os.path.join(DATASET_DIR, image_name)
        image = Image.open(image_path).convert(self.image_format)
        image = self.transform(image)

        return BatchItem(
            image=image,
            labels=label,
            image_fname=image_name,
        )

    def get_labels_presence_for(self, target_label):
        """Returns a list of tuples (idx, 0/1) indicating presence/absence of a
            label for each sample.
        """
        if isinstance(target_label, str):
            target_label = self.labels.index(target_label)

        return [
            (idx, int(target_label == label))
            for idx, (_, label) in enumerate(self.images_metadata)
        ]
