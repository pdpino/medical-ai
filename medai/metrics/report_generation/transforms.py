def get_flat_reports(outputs):
    """Transforms the output to arrays of words indexes.

    Args:
        outputs: dict with keys 'flat_clean_reports_gt' and 'flat_clean_reports_gen'.
            Both are lists of lists with shape (batch_size, n_words_per_report)
    """
    gen = outputs['flat_clean_reports_gen']
    gt = outputs['flat_clean_reports_gt']

    return gen, gt
