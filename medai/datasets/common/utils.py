import torch

from medai.datasets.common.constants import (
    ORGAN_BACKGROUND,
    ORGAN_HEART,
    ORGAN_RIGHT_LUNG,
    ORGAN_LEFT_LUNG,
    CXR14_DISEASES,
    JSRT_ORGANS,
)

# Map from diseases to tuple of organs
_DISEASE_TO_ORGANS = {
    disease: (ORGAN_RIGHT_LUNG, ORGAN_LEFT_LUNG)
    for disease in CXR14_DISEASES
    # Almost all belong to both lungs
}
_DISEASE_TO_ORGANS['Cardiomegaly'] = (ORGAN_HEART,)
_DISEASE_TO_ORGANS['Hernia'] = (ORGAN_BACKGROUND, ORGAN_HEART, ORGAN_RIGHT_LUNG, ORGAN_LEFT_LUNG)


def reduce_masks_for_disease(label, sample_masks, organs=JSRT_ORGANS):
    """Reduce a tensor of organ masks for a given disease.

    Given a tensor of masks for each organ and a disease,
    select only the organs related to the disease and
    collapses the mask into an image.

    Args:
        label -- disease (str)
        sample_masks -- tensor of shape (*, n_organs, height, width)
            Notice it may be masks for a batch (i.e. batch_size is the first dimension),
            or for one sample (i.e. n_organs is the first dimension)
        organs -- list of organs (list of str)

    Returns:
        Mask with organs for the disease, tensor of shape (*, height, width)
    """
    # Get organ names
    if label not in _DISEASE_TO_ORGANS:
        raise KeyError('Disease not in _DISEASE_TO_ORGANS dictionary')
    organs_names = _DISEASE_TO_ORGANS[label]

    # Get organ idxs
    organs_idxs = torch.tensor([ # pylint: disable=not-callable
        organs.index(organ_name)
        for organ_name in organs_names
    ]).to(sample_masks.device)

    # Select organs
    mask = sample_masks.index_select(dim=-3, index=organs_idxs)
    # shape: *, n_selected_organs, height, width

    # Add-up (assume sum wont be more than 1)
    mask = mask.sum(dim=-3)
    # shape: *, height, width

    return mask


def reduce_masks_for_diseases(labels, sample_masks, organs=JSRT_ORGANS):
    """Reduce a tensor of organ masks for multiple diseases.

    i.e. calls reduce_masks_for_disease() multiple times

    Args:
        labels -- diseases (list of str)
        sample_masks -- tensor of shape (*, n_organs, height, width)
            Notice it may be masks for a batch (i.e. batch_size is the first dimension),
            or for one sample (i.e. n_organs is the first dimension)
        organs -- list of organs (list of str)

    Returns:
        Mask with organs for the disease, tensor of shape (*, n_labels, height, width)
    """
    masks = torch.stack([
        reduce_masks_for_disease(label, sample_masks, organs=organs)
        # shape: batch_size, height, width
        for label in labels
    ], dim=-3)

    return masks
