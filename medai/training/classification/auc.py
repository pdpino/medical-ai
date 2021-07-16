import torch

def get_step_fn_cls_auc(model, loss_fn, label_index,
                        optimizer=None, training=True, device='cuda'):
    """Creates a step function for training with AUC loss

    Args:
        model
        loss_fn -- AUCMLoss instance
        label_index -- chosen disease
    """
    def step_fn(unused_engine, data_batch):
        # Move inputs to GPU
        images = data_batch.image.to(device)
        # shape: batch_size, channels=3, height, width

        labels = data_batch.labels.to(device).float()
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

        # Compute classification loss
        cl_loss = loss_fn(outputs[:, label_index], labels[:, label_index])

        if training:
            cl_loss.backward()
            optimizer.step()

        outputs = torch.sigmoid(outputs)

        return {
            'loss': cl_loss.detach(),
            'pred_labels': outputs.detach(),
            'gt_labels': labels,
        }

    return step_fn
