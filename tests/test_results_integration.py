"""Diem noi day chua tung duoc test truoc branch nay: result_dict cua
eval_SAA.py phai song sot qua write_results() thanh cot CSV, va
completed_classes() phai doc lai dung tu CSV do.

metric_cal() (r_f1, p_pro) va summarize_timings() (sau tat cac timing keys +
n_images + peak_vram) da duoc test rieng le. File nay test phan con lai: khi
gop ca hai lai thanh MOT dict roi ghi vao CSV that, khong co key nao roi
mat va khong co class nao bi doc sai trang thai.
"""
import pandas as pd
import pytest

from utils.csv_utils import completed_classes, write_results

# Toan bo key ma pipeline hien tai tao ra trong result_dict (metric_cal +
# summarize_timings), theo dung thu tu lac trong Finding 2 cua review.
FULL_METRICS_KEYS = [
    'i_roc', 'p_roc', 'i_ap', 'p_ap', 'i_f1', 'p_f1', 'r_f1', 'p_pro',
    't_dino', 't_sam', 't_saliency', 't_total', 'n_images', 'peak_vram',
]


def _sample_metrics(offset):
    return {key: float(i + offset) for i, key in enumerate(FULL_METRICS_KEYS)}


def test_write_results_round_trips_every_new_metric_column(tmp_path):
    csv_path = str(tmp_path / 'mvtec-indx-0.csv')
    total_classes = ['carpet', 'grid']

    write_results(_sample_metrics(offset=0), 'carpet', total_classes, csv_path)
    write_results(_sample_metrics(offset=100), 'grid', total_classes, csv_path)

    df = pd.read_csv(csv_path, index_col=0)

    for key in FULL_METRICS_KEYS:
        assert key in df.columns, f'missing column: {key}'

    for i, key in enumerate(FULL_METRICS_KEYS):
        assert df.loc['carpet', key] == pytest.approx(i + 0)
        assert df.loc['grid', key] == pytest.approx(i + 100)

    assert completed_classes(csv_path) == {'carpet', 'grid'}


def test_write_results_upgrades_a_legacy_csv_without_corrupting_old_rows(tmp_path):
    """CSV cu chi co bo cot cu (khong r_f1, khong timing, khong n_images).
    Ghi mot class moi voi bo key day du khong duoc lam hong cac hang cu, va
    cac cot moi tren hang cu phai la blank/NaN chu khong phai rac."""
    csv_path = str(tmp_path / 'mvtec-indx-0.csv')

    pd.DataFrame(
        {
            'i_roc': [88.0, 99.0],
            'p_roc': [77.0, None],
            'i_ap': [66.0, None],
            'p_ap': [55.0, None],
            'i_f1': [44.0, None],
            'p_f1': [33.0, None],
            'p_pro': [22.0, None],
        },
        index=['carpet', 'leather'],
    ).to_csv(csv_path)

    write_results(_sample_metrics(offset=0), 'grid', ['carpet', 'grid'], csv_path)

    df = pd.read_csv(csv_path, index_col=0)

    # Hang cu (carpet) khong bi dam vao cot cu.
    assert df.loc['carpet', 'p_ap'] == pytest.approx(55.0)
    assert df.loc['carpet', 'i_roc'] == pytest.approx(88.0)

    # Hang cu khong co du lieu cho cot moi -> blank/NaN, khong phai gia tri
    # bia dat.
    assert pd.isna(df.loc['carpet', 'r_f1'])
    assert pd.isna(df.loc['carpet', 't_total'])
    assert pd.isna(df.loc['carpet', 'n_images'])

    # Hang moi (grid) co day du moi cot.
    for i, key in enumerate(FULL_METRICS_KEYS):
        assert df.loc['grid', key] == pytest.approx(i + 0)

    # leather chua tung co p_ap (NaN) -> khong duoc coi la done.
    assert pd.isna(df.loc['leather', 'p_ap'])

    done = completed_classes(csv_path)
    assert 'leather' not in done
    assert done == {'carpet', 'grid'}
