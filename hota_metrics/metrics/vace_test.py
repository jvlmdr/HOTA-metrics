import hota_metrics as hm
import numpy as np
import pytest


def test_ata():
    metric = hm.metrics.vace.VACE()
    data = {
            'num_gt_ids': 2,
            'num_tracker_ids': 4,
            'gt_ids': [
                    # t = 0
                    np.array([0, 1], dtype=np.int),
                    # t = 1
                    np.array([0], dtype=np.int),
                    # t = 2
                    np.array([1], dtype=np.int),
            ],
            'tracker_ids': [
                    # t = 0
                    np.array([0, 1, 2], dtype=np.int),
                    # t = 1
                    np.array([1, 2, 3], dtype=np.int),
                    # t = 2
                    np.array([1, 3], dtype=np.int),
            ],
            'similarity_scores': [
                    # t = 0
                    [[0, 1, 1],  # gt 0
                     [1, 1, 1]],  # gt 1
                    # t = 1
                    [[1, 0, 0]],  # gt 0
                    # t = 2
                    [[1, 1]],  # gt 1
            ],
    }
    res = metric.eval_sequence(data)

    # Temporal IOU matrix:
    # pr       0    1    2    3
    # gt 0: [0/2, 2/3, 1/2, 0/3]
    # gt 1: [1/2, 2/3, 1/3, 1/3]
    #
    # Optimal STDA is 2/3 + 1/2 = 7 / 6

    assert res['STDA'] == pytest.approx(7 / 6)
    assert res['ATA_Re'] == pytest.approx((7 / 6) / 2)
    assert res['ATA_Pr'] == pytest.approx((7 / 6) / 4)
    assert res['ATA'] == pytest.approx((7 / 6) / (0.5 * (2 + 4)))
