import collections
import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing


class VACE(_BaseMetric):
    """Class which implements the VACE metrics.

    The metrics are described in:
    Manohar et al. (2006) "Performance Evaluation of Object Detection and Tracking in Video"
    https://link.springer.com/chapter/10.1007/11612704_16

    This implementation uses the "relaxed" variant of the metrics,
    where an overlap threshold is applied in each frame.
    """

    def __init__(self, config=None):
        super().__init__()
        self.integer_fields = ['VACE_IDs', 'VACE_GT_IDs', 'num_non_empty_timesteps']
        self.float_fields = [
            'STDA', 'ATA', 'FDA', 'SFDA',
            'ATR', 'ATP',  # Not defined in original paper.
            # Diagnostics.
            'STDA_gt_sum', 'STDA_gt_max', 'STDA_gt_opt',
            'STDA_pr_sum', 'STDA_pr_max', 'STDA_pr_opt',
            'STDA_gt_error_union_det', 'STDA_gt_error_union_ass',
            'STDA_pr_error_union_det', 'STDA_pr_error_union_ass',
            'ATA_approx', 'ATR_approx', 'ATP_approx',
            'ATR_error_cover_det', 'ATR_error_cover_ass_indep', 'ATR_error_cover_ass_joint',
            'ATR_error_union_det', 'ATR_error_union_ass',
            'ATP_error_cover_det', 'ATP_error_cover_ass_indep', 'ATP_error_cover_ass_joint',
            'ATP_error_union_det', 'ATP_error_union_ass',
            'ATR_error_det_fn', 'ATR_error_ass_split', 'ATR_error_ass_merge',
            'ATP_error_det_fp', 'ATP_error_ass_merge', 'ATP_error_ass_split',
            'ATA_error_det_fn', 'ATA_error_det_fp', 'ATA_error_ass_merge', 'ATA_error_ass_split',
        ]
        self.fields = self.integer_fields + self.float_fields
        self.summary_fields = ['SFDA', 'ATA', 'ATR', 'ATP']

        # Fields that are accumulated over multiple videos.
        self._additive_fields = [
            *self.integer_fields,
            'STDA', 'FDA',
            'STDA_approx',
            'STDA_gt_sum', 'STDA_gt_max', 'STDA_gt_opt',
            'STDA_pr_sum', 'STDA_pr_max', 'STDA_pr_opt',
            'STDA_gt_error_union_det', 'STDA_gt_error_union_ass',
            'STDA_pr_error_union_det', 'STDA_pr_error_union_ass',
        ]

        self.threshold = 0.5

    @_timing.time
    def eval_sequence(self, data):
        """Calculates VACE metrics for one sequence.

        Depends on the fields:
            data['num_gt_ids']
            data['num_tracker_ids']
            data['gt_ids']
            data['tracker_ids']
            data['similarity_scores']
        """
        res = {}

        # Obtain Average Tracking Accuracy (ATA) using track correspondence.
        # Obtain counts necessary to compute temporal IOU.
        # Assume that integer counts can be represented exactly as floats.
        potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        gt_count = np.zeros(data['num_gt_ids'])
        pr_count = np.zeros(data['num_tracker_ids'])
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Count the number of frames in which two tracks satisfy the overlap criterion.
            matches_mask = np.greater_equal(data['similarity_scores'][t], self.threshold)
            match_idx_gt, match_idx_tracker = np.nonzero(matches_mask)
            potential_matches_count[gt_ids_t[match_idx_gt], tracker_ids_t[match_idx_tracker]] += 1
            # Count the number of frames in which the tracks are present.
            gt_count[gt_ids_t] += 1
            pr_count[tracker_ids_t] += 1
        union_count = (gt_count[:, np.newaxis]
                       + pr_count[np.newaxis, :]
                       - potential_matches_count)
        # The denominator should always be non-zero if all tracks are non-empty.
        with np.errstate(divide='raise', invalid='raise'):
            temporal_iou = potential_matches_count / union_count
        # Find assignment that maximizes temporal IOU.
        match_rows, match_cols = linear_sum_assignment(-temporal_iou)
        res['STDA'] = temporal_iou[match_rows, match_cols].sum()
        res['VACE_IDs'] = data['num_tracker_ids']
        res['VACE_GT_IDs'] = data['num_gt_ids']

        # Obtain Frame Detection Accuracy (FDA) using per-frame correspondence.
        non_empty_count = 0
        fda = 0
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            n_g = len(gt_ids_t)
            n_d = len(tracker_ids_t)
            if not (n_g or n_d):
                continue
            # n_g > 0 or n_d > 0
            non_empty_count += 1
            if not (n_g and n_d):
                continue
            # n_g > 0 and n_d > 0
            spatial_overlap = data['similarity_scores'][t]
            match_rows, match_cols = linear_sum_assignment(-spatial_overlap)
            overlap_ratio = spatial_overlap[match_rows, match_cols].sum()
            fda += overlap_ratio / (0.5 * (n_g + n_d))
        res['FDA'] = fda
        res['num_non_empty_timesteps'] = non_empty_count

        # ATA diagnostics.
        match_pairs = _find_match_sequence(data, threshold=self.threshold)
        match_count_dict = collections.Counter(itertools.chain.from_iterable(match_pairs))
        match_count = _to_dense(data['num_gt_ids'], data['num_tracker_ids'], match_count_dict,
                                dtype=int)
        gt_sum_match = np.sum(match_count, axis=1)
        pr_sum_match = np.sum(match_count, axis=0)
        gt_max_match = np.max(match_count, axis=1)
        pr_max_match = np.max(match_count, axis=0)
        res['STDA_gt_sum'] = np.sum(np.true_divide(gt_sum_match, gt_count))
        res['STDA_pr_sum'] = np.sum(np.true_divide(pr_sum_match, pr_count))
        res['STDA_gt_max'] = np.sum(np.true_divide(gt_max_match, gt_count))
        res['STDA_pr_max'] = np.sum(np.true_divide(pr_max_match, pr_count))
        approx_temporal_iou = np.true_divide(match_count, union_count)
        match_rows, match_cols = linear_sum_assignment(-approx_temporal_iou)
        # Exclude empty matches to avoid possible divide by zero.
        not_empty = (match_count[match_rows, match_cols] > 0)
        match_rows = match_rows[not_empty]
        match_cols = match_cols[not_empty]
        res['STDA_approx'] = approx_temporal_iou[match_rows, match_cols].sum()
        opt_match = match_count[match_rows, match_cols]
        opt_union = union_count[match_rows, match_cols]
        gt_frac_opt = np.true_divide(opt_match, gt_count[match_rows])
        pr_frac_opt = np.true_divide(opt_match, pr_count[match_cols])
        res['STDA_gt_opt'] = np.sum(gt_frac_opt)
        res['STDA_pr_opt'] = np.sum(pr_frac_opt)

        res['STDA_gt_error_union_det'] = np.sum(
            gt_frac_opt * ((pr_count[match_cols] - pr_sum_match[match_cols]) / opt_union))
        res['STDA_pr_error_union_det'] = np.sum(
            pr_frac_opt * ((gt_count[match_rows] - gt_sum_match[match_rows]) / opt_union))
        res['STDA_gt_error_union_ass'] = np.sum(
            gt_frac_opt * ((pr_sum_match[match_cols] - opt_match) / opt_union))
        res['STDA_pr_error_union_ass'] = np.sum(
            pr_frac_opt * ((gt_sum_match[match_rows] - opt_match) / opt_union))

        res.update(self._compute_final_fields(res))
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=True):
        """Combines metrics across all classes by averaging over the class values.
        If 'ignore_empty_classes' is True, then it only sums over classes with at least one gt or predicted detection.
        """
        res = {}
        for field in self.fields:
            if ignore_empty_classes:
                res[field] = np.mean([v[field] for v in all_res.values()
                                  if v['VACE_GT_IDs'] > 0 or v['VACE_IDs'] > 0], axis=0)
            else:
                res[field] = np.mean([v[field] for v in all_res.values()], axis=0)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self._additive_fields:
            res[field] = _BaseMetric._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for header in self._additive_fields:
            res[header] = _BaseMetric._combine_sum(all_res, header)
        res.update(self._compute_final_fields(res))
        return res

    @staticmethod
    def _compute_final_fields(additive):
        final = {}
        with np.errstate(invalid='ignore'):  # Permit nan results.
            final['SFDA'] = additive['FDA'] / additive['num_non_empty_timesteps']
            final['ATR'] = additive['STDA'] / additive['VACE_GT_IDs']
            final['ATP'] = additive['STDA'] / additive['VACE_IDs']
            final['ATA'] = (2 * additive['STDA']
                            / (additive['VACE_GT_IDs'] + additive['VACE_IDs']))

            final['ATR_approx'] = additive['STDA_approx'] / additive['VACE_GT_IDs']
            final['ATP_approx'] = additive['STDA_approx'] / additive['VACE_IDs']
            final['ATA_approx'] = (2 * additive['STDA_approx']
                                   / (additive['VACE_GT_IDs'] + additive['VACE_IDs']))

            final['ATR_error_cover_det'] = (
                (additive['VACE_GT_IDs'] - additive['STDA_gt_sum']) / additive['VACE_GT_IDs'])
            final['ATR_error_cover_ass_indep'] = (
                (additive['STDA_gt_sum'] - additive['STDA_gt_max']) / additive['VACE_GT_IDs'])
            final['ATR_error_cover_ass_joint'] = (
                (additive['STDA_gt_max'] - additive['STDA_gt_opt']) / additive['VACE_GT_IDs'])
            final['ATR_error_union_det'] = (
                additive['STDA_gt_error_union_det'] / additive['VACE_GT_IDs'])
            final['ATR_error_union_ass'] = (
                additive['STDA_gt_error_union_ass'] / additive['VACE_GT_IDs'])

            final['ATP_error_cover_det'] = (
                (additive['VACE_IDs'] - additive['STDA_pr_sum']) / additive['VACE_IDs'])
            final['ATP_error_cover_ass_indep'] = (
                (additive['STDA_pr_sum'] - additive['STDA_pr_max']) / additive['VACE_IDs'])
            final['ATP_error_cover_ass_joint'] = (
                (additive['STDA_pr_max'] - additive['STDA_pr_opt']) / additive['VACE_IDs'])
            final['ATP_error_union_det'] = (
                additive['STDA_pr_error_union_det'] / additive['VACE_IDs'])
            final['ATP_error_union_ass'] = (
                additive['STDA_pr_error_union_ass'] / additive['VACE_IDs'])

            final['ATR_error_det_fn'] = final['ATR_error_cover_det']
            final['ATR_error_ass_split'] = final['ATR_error_cover_ass_indep']
            final['ATR_error_ass_merge'] = (final['ATR_error_cover_ass_joint'] +
                                            final['ATR_error_union_ass'])
            final['ATR_error_det_fp'] = final['ATR_error_union_ass']

            final['ATP_error_det_fp'] = final['ATP_error_cover_det']
            final['ATP_error_ass_merge'] = final['ATP_error_cover_ass_indep']
            final['ATP_error_ass_split'] = (final['ATP_error_cover_ass_joint'] +
                                            final['ATP_error_union_ass'])
            final['ATP_error_det_fn'] = final['ATP_error_union_ass']

            final['ATA_error_det_fn'] = (
                ((additive['VACE_GT_IDs'] - additive['STDA_gt_sum']) +
                 additive['STDA_pr_error_union_det'])
                / (additive['VACE_GT_IDs'] + additive['VACE_IDs']))
            final['ATA_error_det_fp'] = (
                ((additive['VACE_IDs'] - additive['STDA_pr_sum']) +
                 additive['STDA_gt_error_union_det'])
                / (additive['VACE_GT_IDs'] + additive['VACE_IDs']))
            final['ATA_error_ass_split'] = (
                ((additive['STDA_gt_sum'] - additive['STDA_gt_max']) +
                 (additive['STDA_pr_max'] - additive['STDA_pr_opt']) +
                 additive['STDA_pr_error_union_ass'])
                / (additive['VACE_GT_IDs'] + additive['VACE_IDs']))
            final['ATA_error_ass_merge'] = (
                ((additive['STDA_pr_sum'] - additive['STDA_pr_max']) +
                 (additive['STDA_gt_max'] - additive['STDA_gt_opt']) +
                 additive['STDA_gt_error_union_ass'])
                / (additive['VACE_GT_IDs'] + additive['VACE_IDs']))

        return final


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
