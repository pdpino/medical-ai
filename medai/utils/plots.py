import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_cm(cm, labels=None, title=None,
            percentage=False, colorbar=False, xlabel=True, ylabel=True):
    """Plots a confusion matrix."""
    if labels is None:
        labels = list(range(len(cm)))

    if isinstance(cm, torch.Tensor):
        cm = cm.detach().numpy()
    if not isinstance(cm, np.ndarray):
        cm = np.array(cm)

    n_labels = len(labels)
    ticks = np.arange(n_labels)

    # pylint: disable=no-member
    plt.imshow(
        cm, interpolation='nearest', cmap=plt.cm.Blues,
    )
    if colorbar:
        plt.colorbar()

    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel('Prediction')
    if ylabel:
        plt.ylabel('True')

    total = cm.sum()

    thresh = cm.max() / 2
    for row in range(n_labels):
        for col in range(n_labels):
            value = cm[row, col]
            color = 'white' if value > thresh else 'black'

            value_str = f'{int(value):d}'
            if percentage:
                value = value / total * 100
                value_str += f'\n({value:.1f}%)'

            plt.text(col, row, value_str, ha='center', va='center', color=color)
