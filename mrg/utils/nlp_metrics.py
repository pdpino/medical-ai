import numpy as np

def indexes_to_strings(candidate, ground_truth):
    """Receives two word-indexes tensors, and returns candidate and gt strings.
    
    Assumes pad_idx is 0, otherwise np.trim_zeros() function would be ugly

    Args:
        candidate -- torch.Tensor of shape n_words
        ground_truth -- torch.Tensor of shape n_words
    Returns:
        candidate_str, ground_truth_strs
        - candidate_str: string of concatenated indexes
        - ground_truth_strs: list of strings of concatenated indexes
    """
    # Trim padding from the end of sentences
    candidate = np.trim_zeros(candidate.cpu().detach().numpy(), 'b')
    ground_truth = np.trim_zeros(ground_truth.cpu().detach().numpy(), 'b')

    # Join as string
    candidate = ' '.join(str(val) for val in candidate)
    ground_truth = ' '.join(str(val) for val in ground_truth)

    # Ground truth is a list of references
    ground_truth = [ground_truth]

    return candidate, ground_truth