
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

    def __init__(self):
        super().__init__()
        self.integer_headers = ['VACE_IDs', 'VACE_GT_IDs']
        self.float_headers = ['STDA', 'ATA', 'ATA_Re', 'ATA_Pr']
        self.headers = self.integer_headers + self.float_headers
        self.summary_headers = ['ATA', 'ATA_Re', 'ATA_Pr']
        self.register_headers_globally()

        # Fields that are accumulated over multiple videos.
        self._additive_headers = self.integer_headers + ['STDA']

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

        # Obtain counts necessary to compute temporal IOU.
        # Assume that integer counts can be represented exactly as floats.
        potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        gt_id_count = np.zeros(data['num_gt_ids'])
        tracker_id_count = np.zeros(data['num_tracker_ids'])
        both_present_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Count the number of frames in which two tracks satisfy the overlap criterion.
            matches_mask = np.greater_equal(data['similarity_scores'][t], self.threshold)
            match_idx_gt, match_idx_tracker = np.nonzero(matches_mask)
            potential_matches_count[gt_ids_t[match_idx_gt], tracker_ids_t[match_idx_tracker]] += 1
            # Count the number of frames in which the tracks are present.
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[tracker_ids_t] += 1
            both_present_count[gt_ids_t[:, None], tracker_ids_t[None, :]] += 1
        # Number of frames in which either track is present (the union of the two sets of frames).
        union_count = (gt_id_count[:, None] + tracker_id_count[None, :] - both_present_count)
        # The denominator should always be non-zero if all tracks are non-empty.
        with np.errstate(divide='raise', invalid='raise'):
            temporal_iou = potential_matches_count / union_count

        # Find assignment that maximizes temporal IOU.
        match_rows, match_cols = linear_sum_assignment(-temporal_iou)

        # Accumulate basic statistics
        res['STDA'] = temporal_iou[match_rows, match_cols].sum()
        res['VACE_IDs'] = data['num_tracker_ids']
        res['VACE_GT_IDs'] = data['num_gt_ids']
        res.update(_compute_final_fields(res))
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for header in self._additive_headers:
            res[header] = _BaseMetric._combine_sum(all_res, header)
        res.update(_compute_final_fields(res))
        return res


def _compute_final_fields(additive):
    final = {}
    final['ATA_Pr'] = additive['STDA'] / additive['VACE_IDs']
    final['ATA_Re'] = additive['STDA'] / additive['VACE_GT_IDs']
    final['ATA'] = 2 / (1 / final['ATA_Re'] + 1 / final['ATA_Pr'])
    return final
