from medai.metrics.segmentation.iou import IoU

def attach_metrics_segmentation(engine, n_labels):
    iou = IoU(n_labels=n_labels)
    iou.attach(engine, 'iou')