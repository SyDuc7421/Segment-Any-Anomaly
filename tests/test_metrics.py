import numpy as np
import pytest

from utils.metrics import calculate_max_f1_region, metric_cal


def _one_square_mask(n_images=2, size=32, top=8, bottom=24):
    """n_images ảnh, mỗi ảnh một hình vuông đặc ở giữa."""
    masks = np.zeros((n_images, size, size), dtype=np.uint8)
    masks[:, top:bottom, top:bottom] = 1
    return masks


def test_max_f1_region_perfect_prediction_is_one():
    gt = _one_square_mask()
    scores = gt.astype(np.float64)

    assert calculate_max_f1_region(gt, scores) == pytest.approx(1.0)


def test_max_f1_region_disjoint_prediction_is_zero():
    gt = _one_square_mask(top=2, bottom=12)

    scores = np.zeros_like(gt, dtype=np.float64)
    scores[:, 20:30, 20:30] = 1.0

    assert calculate_max_f1_region(gt, scores) == pytest.approx(0.0)


def test_metric_cal_reports_r_f1_key_when_pro_disabled():
    gt_mask = _one_square_mask(n_images=4)
    gt_mask[:2] = 0
    scores = gt_mask.astype(np.float64)
    gt_list = np.array([0, 0, 1, 1])

    result = metric_cal(scores, gt_list, list(gt_mask), cal_pro=False)

    assert 'r_f1' in result
    assert result['r_f1'] == 0.0


def test_metric_cal_computes_r_f1_when_pro_enabled(monkeypatch):
    """Tách r_f1 khỏi cal_pro_metric: chỉ kiểm tra phần nối dây, không kiểm tra PRO."""
    import utils.metrics as metrics_module

    monkeypatch.setattr(metrics_module, 'cal_pro_metric', lambda *a, **k: 0.5)

    gt_mask = _one_square_mask(n_images=4)
    gt_mask[:2] = 0
    scores = gt_mask.astype(np.float64)
    gt_list = np.array([0, 0, 1, 1])

    result = metric_cal(scores, gt_list, list(gt_mask), cal_pro=True)

    assert result['r_f1'] == pytest.approx(100.0)
    assert result['p_pro'] == pytest.approx(50.0)
