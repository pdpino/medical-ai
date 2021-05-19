import torch.nn as nn

from medai.models.common import (
    get_adaptive_pooling_layer,
    load_imagenet_model,
)

_ASSERT_IN_OUT_IMAGE_SIZE = False

class ImageNetClsSegModel(nn.Module):
    def __init__(self, cl_labels, seg_labels, model_name='densenet-121',
                 imagenet=True, freeze=False, gpool='max', dropout=0,
                 dropout_features=0,
                 **unused_kwargs):
        super().__init__()
        self.features, self.features_size = load_imagenet_model(
            model_name,
            imagenet=imagenet,
            dropout=dropout,
        )

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        self.cl_labels = list(cl_labels)
        self.seg_labels = list(seg_labels)
        self.model_name = model_name


        self.dropout_features = dropout_features

        # NOTE: this setup works for image input sizes 256, 512, 1024, to output the exact
        # same size in the segmentator (tested on densenet-121, resnet-50, mobilenet).
        # Other input sizes (as 200) may not work
        self.segmentator = nn.Sequential(
            # in: features_size, f-height, f-width
            nn.ConvTranspose2d(self.features_size, 4, 4, 2, padding=1),
            # out: 4, 2x fheight, 2x fwidth
            nn.ConvTranspose2d(4, len(seg_labels), 32, 16, padding=8),
            # out: n_seg_labels, in_size, in_size
        )

        self.cl_reduction = get_adaptive_pooling_layer(gpool, drop=self.dropout_features)

        self.classifier = nn.Linear(self.features_size, len(cl_labels))


    def forward(self, x):
        in_size = x.size()[-2:]

        x = self.features(x)
        # shape: batch_size, n_features, f-height, f-width

        classification = self.classifier(self.cl_reduction(x))
        # shape: batch_size, n_cl_diseases

        segmentation = self.segmentator(x)
        # shape: batch_size, n_seg_labels, height, width

        if _ASSERT_IN_OUT_IMAGE_SIZE:
            out_size = segmentation.size()[-2:]
            assert in_size == out_size, f'Image sizes do not match: in={in_size} vs out={out_size}'

        return classification, segmentation
