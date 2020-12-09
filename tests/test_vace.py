import hota_metrics as hm
import numpy as np
import pytest

metric = hm.metrics.vace.VACE()


def test_vace_2x4():
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
                    np.array([  # t = 0
                            [0, 1, 1],  # gt 0
                            [1, 1, 1],  # gt 1
                    ]),
                    np.array([  # t = 1
                            [1, 0, 0],  # gt 0
                    ]),
                    np.array([  # t = 2
                            [1, 1],  # gt 1
                    ]),
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
    assert res['ATA'] == pytest.approx((7 / 6) / (0.5 * (2 + 4)))

    # Total optimal spatial overlap in each frame:
    # [2, 1, 1]
    # FDA per frame:
    # 2 / ((2 + 3) / 2) = 4 / 5
    # 1 / ((1 + 3) / 2) = 2 / 4 = 1 / 2
    # 1 / ((1 + 2) / 2) = 2 / 3
    # Total FDA:
    # (24 + 15 + 20) / 30 = 59 / 30
    assert res['FDA'] == pytest.approx(59 / 30)
    assert res['VACE_non_empty'] == 3
    assert res['SFDA'] == pytest.approx(59 / 90)


def test_vace_empty():
    num_frames = 3
    data = {
            'num_gt_ids': 0,
            'num_tracker_ids': 0,
            'gt_ids': [np.array([], dtype=np.int) for _ in range(num_frames)],
            'tracker_ids': [np.array([], dtype=np.int) for _ in range(num_frames)],
            'similarity_scores': [np.array([[]]) for _ in range(num_frames)],
    }
    res = metric.eval_sequence(data)
    assert res['STDA'] == 0
    assert np.isnan(res['ATA'])
    assert res['FDA'] == 0
    assert res['VACE_non_empty'] == 0
    assert np.isnan(res['SFDA'])
