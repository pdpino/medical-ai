import math
import logging
import pandas as pd
import numpy as np

LOGGER = logging.getLogger(__name__)

# TODO: reconcile with other classes!!

class HolisticLabeler:
    """Provides a base report-labeler class.

    Subclass this class, and use the labelers with the decorator pattern.
    """
    def __init__(self, labeler):
        self.labeler = labeler

    def __getattr__(self, name):
        return getattr(self.labeler, name)

    def forward(self, reports):
        """Receives a list of reports (str) and returns a np.array of labels.

        Override this method!
        """
        return np.zeros(len(reports), len(self.diseases))

    def __call__(self, reports):
        class_name = self.__class__.__name__
        if not isinstance(reports, (list, np.ndarray)):
            raise Exception(
                f'Expected list or array passed to {class_name}, received: {type(reports)}',
            )

        labels = self.forward(reports)

        expected = (len(reports), len(self.diseases))
        assert labels.shape == expected, \
            f'{class_name} output size do not match: out={labels.shape} vs expect={expected}'

        return labels

    def _assert_out_df_matches_input(self, input_reports, df, key):
        """Asserts an out_df matches the input reports.

        Use this when the labeler uses a DF before returning an array,
        to assert that the output reports are the same and in the same order
        than the input.
        """
        prefix = f'{self.__class__.__name__} i-o mismatch in'

        # Assert sizes
        assert len(df) == len(input_reports), \
            f'{prefix} size: n_input={len(input_reports)} vs n_out={len(df)}'

        # Assert content (i.e. reports inside)
        output_reports = list(df[key])
        set_in = set(input_reports)
        set_out = set(output_reports)
        assert set_in == set_out, \
            f'{prefix} content: in - out={len(set_in - set_out)}, out - in={len(set_out - set_in)}'

        # Assert order
        assert output_reports == list(input_reports), \
            f'{prefix} order: input={input_reports} vs out={output_reports}'


class CacheLookupLabeler(HolisticLabeler):
    """Looks up reports in a cache before labelling them.

    Does not update the cache (only lookup on GT reports).

    Paramters:
        cache -- df with keys ['Reports', *diseases]
    """
    def __init__(self, labeler, cache=None):
        super().__init__(labeler)

        # Remove duplicated Reports, so merge() works properly
        self.cache = cache.groupby('Reports').first().reset_index()

    def forward(self, reports):
        # reports: list of str

        reports_saved_in_cache = set(self.cache['Reports'])

        cached_reports = [
            report for report in reports
            if report in reports_saved_in_cache
        ]

        non_cached_reports = [
            report for report in reports
            if report not in reports_saved_in_cache
        ]

        if len(cached_reports) == 0:
            return self.labeler(reports)

        LOGGER.info(
            'Using GT cache: %s found in cache, %s new ones',
            f'{len(cached_reports):,}',
            f'{len(non_cached_reports):,}'
        )

        df_with_solution = self.cache

        # Label new reports
        if len(non_cached_reports) > 0:
            new_reports_labels = self.labeler(non_cached_reports)

            # Build a DF to use merge()
            new_reports_df = pd.DataFrame(new_reports_labels, columns=self.diseases)
            new_reports_df['Reports'] = non_cached_reports

            df_with_solution = pd.concat([df_with_solution, new_reports_df], axis=0)

        # Target DF with the final result
        target_reports_df = pd.DataFrame(reports, columns=['Reports'])
        target_reports_df['order'] = list(range(len(reports)))
        target_reports_df = target_reports_df.merge(
            df_with_solution, how='left', on='Reports',
        )
        target_reports_df = target_reports_df.sort_values('order')

        self._assert_out_df_matches_input(reports, target_reports_df, 'Reports')

        return target_reports_df[self.diseases].to_numpy()


class NBatchesLabeler(HolisticLabeler):
    """Calls the child labeler splitting the reports in n-batches."""
    LIMIT_PER_BATCH = 10000

    def __init__(self, labeler, batches=None):
        super().__init__(labeler)

        self.n_batches = batches

    def redecide_n_batches(self, n_reports):
        if self.n_batches is None:
            if n_reports > self.LIMIT_PER_BATCH:
                self.n_batches = math.ceil(n_reports / self.LIMIT_PER_BATCH)
            else:
                self.n_batches = 1

    def forward(self, reports):
        # reports: list of str

        n_reports = len(reports)
        self.redecide_n_batches(n_reports)

        if self.n_batches <= 1:
            return self.labeler(reports)

        LOGGER.info(
            'Total reports %s splitted in n_batches=%d of approx %s reports each',
            f'{n_reports:,}', self.n_batches, math.ceil(n_reports / self.n_batches),
        )

        batches = np.array_split(reports, self.n_batches)

        result = np.concatenate([
            self.labeler(batch)
            for batch in batches
            if len(batch) > 0
        ], axis=0)

        return result


class AvoidDuplicatedLabeler(HolisticLabeler):
    """Avoids calculating duplicated results.

    Relies on pd.DataFrame().merge() to find duplicated reports quickly.
    """
    def forward(self, reports):
        # reports: list of str

        # Convert to dataframe, to use merge() later
        aux_df = pd.DataFrame(reports, columns=['text'])
        n_samples = len(aux_df)
        aux_df['order'] = list(range(n_samples))

        # Grab unique reports passed
        unique_reports = aux_df['text'].unique() # np.array of shape: n_unique_reports
        n_unique_reports = len(unique_reports)

        if n_samples == n_unique_reports:
            return self.labeler(reports)

        LOGGER.info(
            'Reduced duplicated reports: from %s to %s unique',
            f'{n_samples:,}',
            f'{n_unique_reports:,}',
        )

        unique_generated = self.labeler(unique_reports)
        # shape: n_unique_reports, n_labels

        expected = (n_unique_reports, len(self.diseases))
        assert unique_generated.shape == expected, \
            f'Wrong shape in duplicated: {unique_generated.shape} vs {expected}'

        df_unique = pd.DataFrame(unique_generated, columns=self.diseases)
        df_unique['unique-reports'] = unique_reports

        aux_df = aux_df.merge(df_unique, how='inner', left_on='text', right_on='unique-reports')
        aux_df = aux_df.sort_values('order')

        self._assert_out_df_matches_input(reports, aux_df, 'text')

        return aux_df[self.diseases].to_numpy()
