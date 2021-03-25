import torch
from torch import nn

def get_step_fn(model, optimizer=None, training=False, multilabel=False,
                loss_weights=None, device='cuda'):
    if isinstance(loss_weights, (list, tuple)):
        loss_weights = torch.tensor(loss_weights, device=device) # pylint: disable=not-callable
    elif isinstance(loss_weights, torch.Tensor):
        loss_weights = loss_weights.to(device)

    if multilabel:
        # REVIEW: use loss_weight as well?
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=loss_weights)

    def step_fn(unused_engine, batch):
        images = batch.image.to(device) # shape: batch_size, 1, height, width
        masks = batch.masks.to(device)
        # shape(multilabel=False): batch_size, height, width
        # shape(multilabel=True): batch_size, n_labels, height, width

        # Enable training
        model.train(training)
        torch.set_grad_enabled(training)

        if training:
            optimizer.zero_grad()

        # Pass thru model
        output = model(images)
        # shape: batch_size, n_labels, height, width

        if multilabel:
            # To work with BCELoss
            masks = masks.float()

        loss = criterion(output, masks)
        batch_loss = loss.item()

        if training:
            loss.backward()
            optimizer.step()

        if multilabel:
            # Metrics assume sigmoided output
            output = torch.sigmoid(output)

        return {
            'loss': batch_loss,
            'activations': output,
            'gt_map': masks,
        }

    return step_fn
