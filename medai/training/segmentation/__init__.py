import torch
from torch import nn

def get_step_fn(model, optimizer=None, training=False,
                loss_weights=None, device='cuda'):
    if isinstance(loss_weights, (list, tuple)):
        loss_weights = torch.tensor(loss_weights).to(device) # pylint: disable=not-callable
    elif isinstance(loss_weights, torch.Tensor):
        loss_weights = loss_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    def step_fn(unused_engine, batch):
        images = batch.image.to(device) # shape: batch_size, 1, height, width
        masks = batch.masks.to(device) # shape: batch_size, height, width

        # Enable training
        model.train(training)
        torch.set_grad_enabled(training)

        if training:
            optimizer.zero_grad()

        # Pass thru model
        output = model(images)
        # shape: batch_size, n_labels, height, width

        loss = criterion(output, masks)
        batch_loss = loss.item()

        if training:
            loss.backward()
            optimizer.step()

        return {
            'loss': batch_loss,
            'activations': output,
            'gt_map': masks,
        }

    return step_fn
