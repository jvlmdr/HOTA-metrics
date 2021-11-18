import collections
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing
from .. import utils


class Identity(_BaseMetric):
    """Class which implements the ID metrics"""

    @staticmethod
    def get_default_config():
        """Default class config values"""
        default_config = {
            'THRESHOLD': 0.5,  # Similarity score threshold required for a IDTP match. Default 0.5.
            'PRINT_CONFIG': True,  # Whether to print the config information on init. Default: False.
            'DIAGNOSTICS': False,  # Whether to include diagnostics. Involves per-frame matching.
        }
        return default_config

    def __init__(self, config=None):
        super().__init__()
        self.config = utils.init_config(config, self.get_default_config(), self.get_name())

        self.integer_fields_base = ['ID_gt_count', 'ID_tracker_count', 'IDTP', 'IDFN', 'IDFP']
        self.integer_fields_diagnostics = ['ID_det', 'ID_gt_max', 'ID_tracker_max', 'IDTP_approx']
        self.float_fields_base = ['IDF1', 'IDR', 'IDP']
        self.float_fields_diagnostics = [
            'IDF1_approx', 'IDR_approx', 'IDP_approx',
            'IDR_error_det_fn', 'IDR_error_ass_split', 'IDR_error_ass_merge',
            'IDP_error_det_fp', 'IDP_error_ass_merge', 'IDP_error_ass_split',
            'IDF1_error_det_fn', 'IDF1_error_det_fp', 'IDF1_error_ass_split', 'IDF1_error_ass_merge',
            'IDR_error_det', 'IDR_error_ass',
            'IDP_error_det', 'IDP_error_ass',
            'IDF1_error_det', 'IDF1_error_ass',
        ]

        self.integer_fields = self.integer_fields_base
        self.float_fields = self.float_fields_base
        if self.config['DIAGNOSTICS']:
            self.integer_fields += self.integer_fields_diagnostics
            self.float_fields += self.float_fields_diagnostics
        self.fields = [*self.float_fields, *self.integer_fields]
        self.summary_fields = ['IDTP', 'IDFN', 'IDFP', *self.float_fields]

        # Configuration options:
        self.threshold = float(self.config['THRESHOLD'])

    @_timing.time
    def eval_sequence(self, data):
        """Calculates ID metrics for one sequence"""
        # Variables counting global association
        gt_count = 0
        tracker_count = 0
        overlap_counts_dict = collections.Counter()

        # First loop through each timestep and accumulate global track information.
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Update the total number of dets.
            gt_count += len(gt_ids_t)
            tracker_count += len(tracker_ids_t)

            # Count the potential matches between ids in each timestep
            overlap_mask = np.greater_equal(data['similarity_scores'][t], self.threshold)
            overlap_idx_gt, overlap_idx_tracker = np.nonzero(overlap_mask)
            # counts_matrix[gt_ids_t[overlap_idx_gt], tracker_ids_t[overlap_idx_tracker]] += 1
            overlap_counts_dict.update(zip(gt_ids_t[overlap_idx_gt],
                                           tracker_ids_t[overlap_idx_tracker]))

        unique_gt_ids = set(gt_id for gt_id, _ in overlap_counts_dict)
        unique_tracker_ids = set(tracker_id for _, tracker_id in overlap_counts_dict)

        match_counts_dict = None
        if self.config['DIAGNOSTICS']:
            matches = _find_match_sequence(data, threshold=self.threshold)
            match_counts_dict = collections.Counter(itertools.chain.from_iterable(matches))

        return self.eval_sequence_from_counts(
            gt_count, tracker_count,
            max(unique_gt_ids) + 1, max(unique_tracker_ids) + 1,
            overlap_counts_dict, match_counts_dict)

    @_timing.time
    def eval_sequence_from_counts(self,
                                  gt_count, tracker_count,
                                  num_gt_ids, num_tracker_ids,
                                  overlap_counts_dict, match_counts_dict=None):
        """Calculates confusion metrics for one sequence"""
        res = {}

        res['ID_gt_count'] = gt_count
        res['ID_tracker_count'] = tracker_count
        # Find maximum one-to-one correspondence.
        overlap_counts = _to_dense(num_gt_ids, num_tracker_ids, overlap_counts_dict, dtype=int)
        match_rows, match_cols = linear_sum_assignment(-overlap_counts.astype(float))
        res['IDTP'] = overlap_counts[match_rows, match_cols].sum()
        # Legacy fields.
        res['IDFN'] = res['ID_gt_count'] - res['IDTP']
        res['IDFP'] = res['ID_tracker_count'] - res['IDTP']

        if self.config['DIAGNOSTICS']:
            match_counts = _to_dense(num_gt_ids, num_tracker_ids, match_counts_dict, dtype=int)
            res['ID_det'] = np.sum(match_counts)
            res['ID_gt_max'] = np.sum(np.max(match_counts, axis=1))
            res['ID_tracker_max'] = np.sum(np.max(match_counts, axis=0))
            match_rows, match_cols = linear_sum_assignment(-match_counts.astype(float))
            res['IDTP_approx'] = match_counts[match_rows, match_cols].sum()

        res = self._compute_final_fields(res)
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values.
        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        res = {}
        for field in self.integer_fields:
            if ignore_empty_classes:
                res[field] = self._combine_sum({k: v for k, v in all_res.items()
                                                if v['IDTP'] + v['IDFN'] + v['IDFP'] > 0 + np.finfo('float').eps},
                                               field)
            else:
                res[field] = self._combine_sum({k: v for k, v in all_res.items()}, field)
        for field in self.float_fields:
            if ignore_empty_classes:
                res[field] = np.mean([v[field] for v in all_res.values()
                                      if v['IDTP'] + v['IDFN'] + v['IDFP'] > 0 + np.finfo('float').eps], axis=0)
            else:
                res[field] = np.mean([v[field] for v in all_res.values()], axis=0)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    def _compute_final_fields(self, res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        res = dict(res)
        res['IDR'] = res['IDTP'] / res['ID_gt_count']
        res['IDP'] = res['IDTP'] / res['ID_tracker_count']
        res['IDF1'] = res['IDTP'] / (0.5 * (res['ID_gt_count'] + res['ID_tracker_count']))

        if self.config['DIAGNOSTICS']:
            res['IDR_approx'] = res['IDTP_approx'] / res['ID_gt_count']
            res['IDP_approx'] = res['IDTP_approx'] / res['ID_tracker_count']
            res['IDF1_approx'] = res['IDTP_approx'] / (0.5 * (res['ID_gt_count'] + res['ID_tracker_count']))
            res['IDR_error_det_fn'] = (res['ID_gt_count'] - res['ID_det']) / res['ID_gt_count']
            res['IDR_error_ass_split'] = (res['ID_det'] - res['ID_gt_max']) / res['ID_gt_count']
            res['IDR_error_ass_merge'] = (res['ID_gt_max'] - res['IDTP_approx']) / res['ID_gt_count']
            res['IDP_error_det_fp'] = (res['ID_tracker_count'] - res['ID_det']) / res['ID_tracker_count']
            res['IDP_error_ass_merge'] = (res['ID_det'] - res['ID_tracker_max']) / res['ID_tracker_count']
            res['IDP_error_ass_split'] = (res['ID_tracker_max'] - res['IDTP_approx']) / res['ID_tracker_count']
            res['IDF1_error_det_fn'] = (
                (res['ID_gt_count'] - res['ID_det']) /
                (res['ID_gt_count'] + res['ID_tracker_count']))
            res['IDF1_error_det_fp'] = (
                (res['ID_tracker_count'] - res['ID_det']) /
                (res['ID_gt_count'] + res['ID_tracker_count']))
            res['IDF1_error_ass_split'] = (
                ((res['ID_det'] - res['ID_gt_max']) + (res['ID_tracker_max'] - res['IDTP_approx'])) /
                (res['ID_gt_count'] + res['ID_tracker_count']))
            res['IDF1_error_ass_merge'] = (
                ((res['ID_det'] - res['ID_tracker_max']) + (res['ID_gt_max'] - res['IDTP_approx'])) /
                (res['ID_gt_count'] + res['ID_tracker_count']))
            res['IDR_error_det'] = res['IDR_error_det_fn']
            res['IDR_error_ass'] = res['IDR_error_ass_split'] + res['IDR_error_ass_merge']
            res['IDP_error_det'] = res['IDP_error_det_fp']
            res['IDP_error_ass'] = res['IDP_error_ass_merge'] + res['IDP_error_ass_split']
            res['IDF1_error_det'] = res['IDF1_error_det_fn'] + res['IDF1_error_det_fp']
            res['IDF1_error_ass'] = res['IDF1_error_ass_split'] + res['IDF1_error_ass_merge']

        return res


# TODO: Avoid duplication in metrics/confusion.py.
def _find_match_sequence(data, threshold):
    """Returns list of id pairs for each frame."""
    matches = [[] for _ in range(data['num_timesteps'])]
    for t, (gt_ids, pr_ids) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
        if len(gt_ids) == 0:
            continue
        if len(pr_ids) == 0:
            continue
        # Construct score matrix to optimize number of matches and then localization
        similarity = data['similarity_scores'][t]
        assert np.all(~(similarity < 0))
        assert np.all(~(similarity > 1))
        eps = 1. / (max(similarity.shape) + 1.)
        is_overlap = (similarity >= threshold)
        similarity = np.where(is_overlap, similarity, 0.)
        score_mat = is_overlap.astype(float) + eps * similarity
        # Hungarian algorithm to find best assignment
        opt_rows, opt_cols = linear_sum_assignment(-score_mat)
        # Exclude pairs that do not satisfy criterion
        pair_is_valid = is_overlap[opt_rows, opt_cols]
        num_matches = np.count_nonzero(pair_is_valid)
        match_gt_ids = gt_ids[opt_rows[pair_is_valid]]
        match_pr_ids = pr_ids[opt_cols[pair_is_valid]]
        # Ensure that similarity could not have overwhelmed a match.
        delta = np.sum(score_mat[opt_rows, opt_cols]) - num_matches
        assert 0 <= delta
        assert delta < 1
        # Add (t, gt, pr) tuples to match list.
        matches[t] = list(zip(match_gt_ids, match_pr_ids))
    return matches


def _to_dense(m, n, values, dtype=None):
    x = np.zeros((m, n), dtype=dtype)
    for k, v in values.items():
        x[k] = v
    return x
