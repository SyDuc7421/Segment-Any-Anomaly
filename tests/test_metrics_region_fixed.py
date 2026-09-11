"""max-F1-region cai dung dinh nghia trong paper.

Paper (docs/SAA+.md dong 332-337): "compute the F1-score for region-wise
segmentation at the optimal threshold, considering a prediction positive if
the overlapping value exceeds 0.6". F1 theo dinh nghia do luon <= 1.
"""

import numpy as np
import pytest

from utils.metrics import calculate_max_f1_region, calculate_max_f1_region_fixed


def _one_square(size=60, lo=20, hi=40):
    gt = np.zeros((1, size, size), dtype=np.uint8)
    gt[0, lo:hi, lo:hi] = 1
    return gt


def test_perfect_prediction_is_one():
    gt = _one_square()
    scores = gt.astype(np.float64)

    assert calculate_max_f1_region_fixed(gt, scores) == pytest.approx(1.0)


def test_disjoint_prediction_is_zero():
    gt = _one_square(lo=5, hi=20)

    scores = np.zeros_like(gt, dtype=np.float64)
    scores[0, 40:55, 40:55] = 1.0

    assert calculate_max_f1_region_fixed(gt, scores) == pytest.approx(0.0)


def test_fragmented_prediction_never_exceeds_one():
    """Hai manh du doan cung trum mot vung GT.

    Ban cu tra ve 1.333: no dem vung DU DOAN khop duoc roi chia cho so vung
    GT, nen recall = 2/1. Ban fixed ghep mot-mot nen recall <= 1.
    """
    gt = _one_square()

    scores = np.zeros((1, 60, 60), dtype=np.float64)
    scores[0, 20:40, 20:40] = 1.0
    scores[0, 29:31, 20:40] = 0.0     # khe doc tach du doan thanh 2 manh

    legacy = calculate_max_f1_region(gt, scores)
    fixed = calculate_max_f1_region_fixed(gt, scores)

    assert legacy > 1.0, 'fixture nay phai kich hoat duoc loi cua ban cu'
    assert fixed <= 1.0


def test_many_predictions_one_gt_caps_recall():
    """Nam manh du doan, mot vung GT. recall khong the vuot 1."""
    gt = _one_square()

    scores = np.zeros((1, 60, 60), dtype=np.float64)
    scores[0, 20:40, 20:40] = 1.0
    for gap in (23, 27, 31, 35):
        scores[0, gap:gap + 1, 20:40] = 0.0

    assert calculate_max_f1_region_fixed(gt, scores) <= 1.0


def test_two_regions_matched_one_to_one():
    """Hai vung GT, hai vung du doan trung khop -> F1 = 1."""
    gt = np.zeros((1, 60, 60), dtype=np.uint8)
    gt[0, 5:20, 5:20] = 1
    gt[0, 40:55, 40:55] = 1

    scores = gt.astype(np.float64)

    assert calculate_max_f1_region_fixed(gt, scores) == pytest.approx(1.0)


def test_half_the_gt_regions_found():
    """Hai vung GT, du doan chi trung mot -> precision 1, recall 0.5, F1 = 2/3."""
    gt = np.zeros((1, 60, 60), dtype=np.uint8)
    gt[0, 5:20, 5:20] = 1
    gt[0, 40:55, 40:55] = 1

    scores = np.zeros((1, 60, 60), dtype=np.float64)
    scores[0, 5:20, 5:20] = 1.0

    assert calculate_max_f1_region_fixed(gt, scores) == pytest.approx(2 / 3)


def test_overlap_below_threshold_does_not_count():
    """IoU 0.25 < 0.6 -> khong tinh la khop."""
    gt = _one_square(lo=20, hi=40)          # 20x20

    scores = np.zeros((1, 60, 60), dtype=np.float64)
    scores[0, 20:30, 20:30] = 1.0           # 10x10 nam gon trong GT, IoU = 0.25

    assert calculate_max_f1_region_fixed(gt, scores) == pytest.approx(0.0)


def test_metric_cal_reports_r_f1_fixed(monkeypatch):
    import utils.metrics as metrics_module

    monkeypatch.setattr(metrics_module, 'cal_pro_metric', lambda *a, **k: 0.5)

    gt_mask = np.zeros((4, 32, 32), dtype=np.uint8)
    gt_mask[2:, 8:24, 8:24] = 1
    scores = gt_mask.astype(np.float64)
    gt_list = np.array([0, 0, 1, 1])

    result = metrics_module.metric_cal(scores, gt_list, list(gt_mask), cal_pro=True)

    assert result['r_f1_fixed'] == pytest.approx(100.0)


def test_metric_cal_has_r_f1_fixed_key_when_pro_disabled():
    from utils.metrics import metric_cal

    gt_mask = np.zeros((4, 32, 32), dtype=np.uint8)
    gt_mask[2:, 8:24, 8:24] = 1
    scores = gt_mask.astype(np.float64)
    gt_list = np.array([0, 0, 1, 1])

    result = metric_cal(scores, gt_list, list(gt_mask), cal_pro=False)

    assert result['r_f1_fixed'] == 0.0
