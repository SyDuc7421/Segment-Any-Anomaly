import pandas as pd
import pytest

from utils.csv_utils import check_run_identity, completed_classes


@pytest.fixture
def csv_path(tmp_path):
    return str(tmp_path / 'mvtec-indx-0.csv')


def test_missing_file_means_nothing_is_done(csv_path):
    assert completed_classes(csv_path) == set()


def test_rows_with_a_positive_metric_count_as_done(csv_path):
    pd.DataFrame(
        {'p_ap': [12.5, 0.0, 3.0]},
        index=['carpet', 'grid', 'leather'],
    ).to_csv(csv_path)

    assert completed_classes(csv_path) == {'carpet', 'leather'}


def test_placeholder_rows_are_not_done(csv_path):
    """write_results khoi tao moi class bang 0.00 truoc khi chay."""
    pd.DataFrame(
        {'p_ap': [0.0, 0.0]},
        index=['carpet', 'grid'],
    ).to_csv(csv_path)

    assert completed_classes(csv_path) == set()


def test_missing_metric_column_means_nothing_is_done(csv_path):
    pd.DataFrame({'i_roc': [90.0]}, index=['carpet']).to_csv(csv_path)

    assert completed_classes(csv_path) == set()


def test_visa_rows_keep_their_dataset_prefix(csv_path):
    """save_metric ghi 'visa_public-candle' cho dataset khac mvtec."""
    pd.DataFrame(
        {'p_ap': [5.0]},
        index=['visa_public-candle'],
    ).to_csv(csv_path)

    assert completed_classes(csv_path) == {'visa_public-candle'}


IDENTITY = {
    'max_samples': None,
    'cal_pro': False,
    'sam_variant': 'vit_h',
    'saliency_backbone': 'wide_resnet50',
}


def test_check_run_identity_matches_when_every_field_agrees():
    meta = {'mvtec-carpet': dict(IDENTITY)}

    ok, reason = check_run_identity(meta, 'mvtec-carpet', IDENTITY)

    assert ok is True
    assert reason == ''


def test_check_run_identity_fails_when_meta_has_no_entry_for_the_class():
    """CSV cu (legacy) hoac class chua tung duoc ghi run_meta.json."""
    meta = {}

    ok, reason = check_run_identity(meta, 'mvtec-carpet', IDENTITY)

    assert ok is False
    assert reason != ''


def test_check_run_identity_flags_a_max_samples_mismatch():
    """Day la ca smoke-test-34-anh-roi-full-run-117-anh cua Finding 1."""
    meta = {'mvtec-carpet': dict(IDENTITY, max_samples=34)}

    ok, reason = check_run_identity(meta, 'mvtec-carpet', IDENTITY)

    assert ok is False
    assert 'max_samples' in reason


def test_check_run_identity_flags_a_sam_variant_mismatch():
    meta = {'mvtec-carpet': dict(IDENTITY, sam_variant='mobile_sam')}

    ok, reason = check_run_identity(meta, 'mvtec-carpet', IDENTITY)

    assert ok is False
    assert 'sam_variant' in reason


def test_check_run_identity_flags_a_saliency_backbone_mismatch():
    meta = {'mvtec-carpet': dict(IDENTITY, saliency_backbone='mobilenetv3')}

    ok, reason = check_run_identity(meta, 'mvtec-carpet', IDENTITY)

    assert ok is False
    assert 'saliency_backbone' in reason


def test_check_run_identity_flags_a_cal_pro_mismatch():
    meta = {'mvtec-carpet': dict(IDENTITY, cal_pro=True)}

    ok, reason = check_run_identity(meta, 'mvtec-carpet', IDENTITY)

    assert ok is False
    assert 'cal_pro' in reason


def test_check_run_identity_flags_a_missing_field_in_the_stored_entry():
    """Entry cu thieu han mot truong (vd ghi truoc khi them sam_variant)."""
    entry = dict(IDENTITY)
    del entry['sam_variant']
    meta = {'mvtec-carpet': entry}

    ok, reason = check_run_identity(meta, 'mvtec-carpet', IDENTITY)

    assert ok is False
    assert 'sam_variant' in reason
