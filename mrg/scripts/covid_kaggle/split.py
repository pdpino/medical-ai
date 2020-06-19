import random
import os

from mrg.datasets.covid_kaggle import DATASET_DIR, _FOLDERS

def save_list(items, name):
    filepath = os.path.join(DATASET_DIR, f'{name}.txt')
    with open(filepath, 'w') as f:
        for item in items:
            f.write(f'{item}\n')

    print(f'List saved to: {filepath}')


def main(val_split=0.1, test_split=0.1):
    all_val_images = []
    all_test_images = []
    all_train_images = []

    for label in _FOLDERS:
        folder = os.path.join(DATASET_DIR, label)
        images = os.listdir(folder)

        n_images = len(images)
        n_val = int(val_split * n_images)
        n_test = int(val_split * n_images)

        val_test_images = random.sample(images, (n_val + n_test))

        val_images = val_test_images[:n_val]
        test_images = val_test_images[n_val:]
        train_images = [name for name in images if name not in val_test_images]

        all_val_images.extend(val_images)
        all_test_images.extend(test_images)
        all_train_images.extend(train_images)

    save_list(all_train_images, 'train')
    save_list(all_val_images, 'val')
    save_list(all_test_images, 'test')


if __name__ == '__main__':
    main()