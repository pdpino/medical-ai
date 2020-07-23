import matplotlib.pyplot as plt
import numpy as np


def plot_cm(cm, labels=None, title=None, percentage=False, colorbar=False):
    if labels is None:
        labels = list(range(len(cm)))

    if not isinstance(cm, np.ndarray):
        cm = np.array(cm)

    n_labels = len(labels)
    ticks = np.arange(n_labels)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    if colorbar: plt.colorbar()

    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    if title: plt.title(title)
    plt.xlabel('Prediction')
    plt.ylabel('True')

    total = cm.sum()
    
    thresh = cm.max() / 2
    for row in range(n_labels):
        for col in range(n_labels):
            value = cm[row, col]
            color = 'white' if value > thresh else 'black'
            
            value_str = f'{int(value):d}'
            if percentage:
                value_str += f'\n({value/total*100:.1f}%)'

            plt.text(col, row, value_str, ha='center', va='center', color=color)