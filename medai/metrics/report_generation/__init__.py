import operator
import logging
import numpy as np
from torch.nn.functional import interpolate
from ignite.metrics import RunningAverage, MetricsLambda

from medai.metrics import attach_losses
from medai.metrics.segmentation.iou import IoU
from medai.metrics.segmentation.iobb import IoBB
from medai.metrics.report_generation.word_accuracy import WordAccuracy
from medai.metrics.report_generation.nlp.bleu import Bleu
from medai.metrics.report_generation.nlp.rouge import RougeL
from medai.metrics.report_generation.nlp.cider import CiderD
from medai.metrics.report_generation.distinct_sentences import DistinctSentences
from medai.metrics.report_generation.distinct_words import DistinctWords
from medai.metrics.report_generation.transforms import get_flat_reports
from medai.metrics.report_generation.organ_by_sentence import OrganBySentence

LOGGER = logging.getLogger(__name__)


def _attach_bleu(engine, up_to_n=4,
                 output_transform=get_flat_reports,
                 device='cuda',
                 ):
    bleu_up_to_n = Bleu(n=up_to_n, output_transform=output_transform, device=device)
    for i in range(up_to_n):
        bleu_n = MetricsLambda(operator.itemgetter(i), bleu_up_to_n)
        bleu_n.attach(engine, f'bleu{i+1}')

    bleu_avg = MetricsLambda(np.mean, bleu_up_to_n)
    bleu_avg.attach(engine, 'bleu')


def attach_attention_vs_masks(engine, device='cuda'):
    """Attaches metrics that evaluate attention scores vs gt-masks."""
    def _get_masks_and_attention(outputs):
        """Extracts generated and GT masks.

        Args:
            outputs: dict with tensors:
                ['gen_masks']: shape batch_size, n_sentences, features-height, features-width
                ['gt_masks']: shape batch_size, n_sentences, original-height, original-width
                ['gt_stops']: shape batch_size, n_sentences (optional)

            Notice features-* sizes will probably be smaller than original-* sizes,
            as the former are extracted from the last layer of a CNN,
            and the latter are the original GT masks.

        Returns:
            tuple with two (optional three) tensors
        """
        gen_masks = outputs['gen_masks']
        gt_masks = outputs['gt_masks']

        # Reduce gt_masks to gen_masks size
        features_dimensions = gen_masks.size()[-2:]
        gt_masks = interpolate(gt_masks.float(), features_dimensions, mode='nearest').long()

        # Include stops if present
        if 'gt_stops' in outputs:
            gt_stops = outputs['gt_stops']

            # Transform stops into valid
            # stop == 1 indicates stopping --> sentence not valid --> valid == 0
            # stop == 0 indicates dont-stop --> sentence valid --> valid == 1
            gt_valid = 1 - gt_stops
        else:
            gt_valid = None

        return gen_masks, gt_masks, gt_valid

    metrics_kwargs = {
        'reduce_sum': True,
        'output_transform': _get_masks_and_attention,
        'device': device,
    }

    iou = IoU(**metrics_kwargs)
    iou.attach(engine, 'att_iou')

    iobb = IoBB(**metrics_kwargs)
    iobb.attach(engine, 'att_iobb')


def attach_metrics_report_generation(engine, free=False, device='cuda'):
    metric_kwargs = {
        'output_transform': get_flat_reports,
        'device': device,
    }

    # Attach word accuracy
    if not free:
        pass
        # word_acc = WordAccuracy(**metric_kwargs)
        # word_acc.attach(engine, 'word_acc')

    # Attach multiple bleu
    _attach_bleu(engine, 4, **metric_kwargs)

    rouge = RougeL(**metric_kwargs)
    rouge.attach(engine, 'rougeL')

    cider = CiderD(**metric_kwargs)
    cider.attach(engine, 'ciderD')

    # Attach variability
    distinct_words = DistinctWords(**metric_kwargs)
    distinct_words.attach(engine, 'distinct_words')

    distinct_sentences = DistinctSentences(**metric_kwargs)
    distinct_sentences.attach(engine, 'distinct_sentences')


def attach_losses_rg(engine, free=False, hierarchical=False,
                     supervise_attention=False, supervise_sentences=False,
                     device='cuda'):
    if free:
        return

    losses = []
    if hierarchical:
        losses.extend(['word_loss', 'stop_loss'])
    if supervise_attention:
        losses.append('att_loss')
    if supervise_sentences:
        losses.append('sentence_loss')
    attach_losses(engine, losses, device=device)


def attach_organ_by_sentence(engine, vocab, should_attach=True, device='cuda'):
    if not should_attach:
        return

    metric = OrganBySentence(vocab, output_transform=get_flat_reports, device=device)
    metric.attach(engine, 'organ-acc')
