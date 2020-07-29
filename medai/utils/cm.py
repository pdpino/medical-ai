import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_cm(cm, classes, title='', percentage=False, colorbar=False):
    """Plots a confusion matrix."""
    if isinstance(cm, torch.Tensor):
        cm = cm.numpy()

    n_classes = len(classes)
    ticks = np.arange(n_classes)

    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    if colorbar:
        plt.colorbar()

    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)
    plt.xlabel('Prediction')
    plt.ylabel('True')
    if title:
        plt.title(title)

    total = cm.sum()
    
    thresh = cm.max() / 2
    for row in range(n_classes):
        for col in range(n_classes):
            value = cm[row, col]
            color = 'white' if value > thresh else 'black'
            
            value_str = f'{int(value):d}'
            if percentage:
                value /= total
                value_str += f'\n({value:.3f})'
                
            plt.text(col, row, value_str, ha='center', va='center', color=color)
