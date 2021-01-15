import torch

def get_step_fn(model, loss_fn, optimizer=None, training=True, multilabel=True, device='cuda'):
    """Creates a step function for an Engine."""
    def step_fn(unused_engine, data_batch):
        # Move inputs to GPU
        images = data_batch.image.to(device)
        # shape: batch_size, channels=3, height, width

        labels = data_batch.labels.to(device)
        # shape(multilabel=True): batch_size, n_labels
        # shape(multilabel=False): batch_size

        # Enable training
        model.train(training)
        torch.set_grad_enabled(training)

        # zero the parameter gradients
        if training:
            optimizer.zero_grad()

        # Forward
        output_tuple = model(images)
        outputs = output_tuple[0]
        # shape: batch_size, n_labels

        if multilabel:
            labels = labels.float()
        else:
            labels = labels.long()

        # Compute classification loss
        loss = loss_fn(outputs, labels)

        batch_loss = loss.item()

        if training:
            loss.backward()
            optimizer.step()

        if multilabel:
            # NOTE: multilabel metrics assume output is sigmoided
            outputs = torch.sigmoid(outputs)

        return batch_loss, outputs, labels

    return step_fn
