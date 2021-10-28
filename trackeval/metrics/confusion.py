import collections
import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._base_metric import _BaseMetric
from .. import _timing
from .. import utils

_METRICS_OVER_TP = ['sqr', 'rms', 'max']
_METRICS_WITH_DIVISOR = ['ent', 'rand']


class Confusion(_BaseMetric):
    """Class which implements various confusion metrics."""

    @staticmethod
    def get_default_config():
        """Default class config values"""
        default_config = {
            'THRESHOLD': 0.5,  # Similarity score threshold required for a match. Default 0.5.
            'PRINT_CONFIG': True,  # Whether to print the config information on init. Default: False.
        }
        return default_config

    def __init__(self, config=None):
        super().__init__()
        self.integer_fields = ['Ass_present']
        self.float_fields = sum([
            ['AssJ_correct', 'AssJ'],
            *[[f'AssRe_correct_{v}', f'AssPr_correct_{v}']
              for v in _METRICS_OVER_TP + _METRICS_WITH_DIVISOR],
            *[[f'AssRe_present_{v}', f'AssPr_present_{v}']
              for v in _METRICS_WITH_DIVISOR],
            *[[f'AssRe_{v}', f'AssPr_{v}', f'AssF1_{v}']
              for v in _METRICS_OVER_TP + _METRICS_WITH_DIVISOR],
        ], [])
        self.fields = self.float_fields + self.integer_fields
        self.summary_fields = sum([
            ['AssJ'],
            *[[f'AssRe_{v}', f'AssPr_{v}', f'AssF1_{v}']
              for v in _METRICS_OVER_TP + _METRICS_WITH_DIVISOR],
        ], [])
        self.summed_fields = sum([
            ['Ass_present', 'AssJ_correct'],
            *[[f'AssRe_correct_{v}', f'AssPr_correct_{v}']
              for v in _METRICS_OVER_TP + _METRICS_WITH_DIVISOR],
            *[[f'AssRe_present_{v}', f'AssPr_present_{v}']
              for v in _METRICS_WITH_DIVISOR],
        ], [])

        # Configuration options:
        self.config = utils.init_config(config, self.get_default_config(), self.get_name())
        self.threshold = float(self.config['THRESHOLD'])

    @_timing.time
    def eval_sequence(self, data):
        """Calculates confusion metrics for one sequence"""
        matches = _find_match_sequence(data, threshold=self.threshold)
        counts_dict = collections.Counter(itertools.chain.from_iterable(matches))
        counts = _to_dense(data['num_gt_ids'], data['num_tracker_ids'], counts_dict, dtype=float)

        with np.errstate(all='raise'):
            res = {}
            res['Ass_present'] = sum(counts_dict.values())
            for v in _METRICS_OVER_TP:
                correct_fn = _CORRECT_FNS[v]
                res[f'AssRe_correct_{v}'] = correct_fn(counts, axis=1)
                res[f'AssPr_correct_{v}'] = correct_fn(counts, axis=0)
            for v in _METRICS_WITH_DIVISOR:
                correct_fn = _CORRECT_FNS[v]
                res[f'AssRe_correct_{v}'], res[f'AssRe_present_{v}'] = correct_fn(counts, axis=1)
                res[f'AssPr_correct_{v}'], res[f'AssPr_present_{v}'] = correct_fn(counts, axis=0)
            res['AssJ_correct'] = correct_jac(counts)

        res = self._compute_final_fields(res)
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        """Combines metrics across all classes by averaging over the class values.

        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        raise NotImplementedError()

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        raise NotImplementedError()

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.summed_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.

        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        res['AssJ'] = res['AssJ_correct'] / res['Ass_present']
        for v in _METRICS_OVER_TP:
            res[f'AssRe_{v}'] = res[f'AssRe_correct_{v}'] / res['Ass_present']
            res[f'AssPr_{v}'] = res[f'AssPr_correct_{v}'] / res['Ass_present']
        for v in _METRICS_WITH_DIVISOR:
            res[f'AssRe_{v}'] = res[f'AssRe_correct_{v}'] / res[f'AssRe_present_{v}']
            res[f'AssPr_{v}'] = res[f'AssPr_correct_{v}'] / res[f'AssPr_present_{v}']
        for v in _METRICS_OVER_TP + _METRICS_WITH_DIVISOR:
            res[f'AssF1_{v}'] = _harmean(res[f'AssRe_{v}'], res[f'AssPr_{v}'])
        return res


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


def correct_max(counts, axis):
    correct = np.max(counts, axis=axis)
    return np.sum(correct)


def correct_sqr(counts, axis):
    totals = np.sum(counts, axis=axis, keepdims=True)
    with np.errstate(invalid='ignore'):
        frac = np.sum(np.true_divide(counts, totals)**2, axis=axis, keepdims=True)
    correct = totals * np.where(totals == 0, 0., frac)
    assert np.all(np.sum(correct) <= np.sum(counts))
    return np.sum(correct)


def correct_rms(counts, axis):
    totals = np.sum(counts, axis=axis, keepdims=True)
    with np.errstate(invalid='ignore'):
        frac = np.linalg.norm(np.true_divide(counts, totals), axis=axis, keepdims=True)
    correct = totals * np.where(totals == 0, 0., frac)
    assert np.all(np.sum(correct) <= np.sum(counts))
    return np.sum(correct)


def correct_pair(counts, axis):
    totals = np.sum(counts, axis=axis, keepdims=True)
    with np.errstate(invalid='ignore'):
        frac = np.sum(np.true_divide(counts * (counts - 1), totals * (totals - 1)),
                      axis=axis, keepdims=True)
    correct = totals * np.where(totals == 0, 0., frac)
    assert np.all(np.sum(correct) <= np.sum(counts))
    return np.sum(correct)


def correct_ent(counts, axis):
    totals = np.sum(counts, axis=axis, keepdims=True)
    with np.errstate(invalid='ignore'):
        cond_prob = np.where(totals == 0, 0., counts / totals)
    cond_ent = np.sum(np.where(totals == 0, 0., _xlogx(cond_prob)),
                      axis=axis, keepdims=True)
    # Take expectation of entropy over reference tracks.
    num_tp = np.sum(totals)
    prob = totals / num_tp
    expected_cond_ent = np.sum(np.where(totals == 0, 0., prob * cond_ent))
    # Compare entropy within track to reference entropy.
    ref_axis = 1 - axis
    ref_totals = np.sum(counts, axis=ref_axis, keepdims=False)
    ref_prob = _xlogx(ref_totals / num_tp)
    ref_ent = 0. if num_tp == 0 else np.sum(_xlogx(ref_prob))
    return ref_ent - expected_cond_ent, ref_ent


def correct_rand(counts, axis):
    totals = np.sum(counts, axis=axis)
    correct = np.where(counts > 1, counts * (counts - 1), 0)
    present = np.where(totals > 1, totals * (totals - 1), 0)
    return np.sum(correct), np.sum(present)


def _xlogx(p):
    with np.errstate(invalid='ignore', divide='ignore'):
        return p * np.where(p == 0, 0., -np.log(p))


def correct_jac(counts):
    row_totals = np.sum(counts, axis=1, keepdims=True)
    col_totals = np.sum(counts, axis=0, keepdims=True)
    with np.errstate(invalid='ignore'):
        frac = np.true_divide(counts, row_totals + col_totals - counts)
    correct = counts * np.where(counts == 0, 0., frac)
    assert np.all(np.sum(correct) <= np.sum(counts))
    return np.sum(correct)


_CORRECT_FNS = {
        'max': correct_max,
        'sqr': correct_sqr,
        'rms': correct_rms,
        'ent': correct_ent,
        'rand': correct_rand,
        # 'pair': correct_pair,
}


def _harmean(a, b):
    return np.true_divide(1, 0.5 * (np.true_divide(1, a) + np.true_divide(1, b)))
