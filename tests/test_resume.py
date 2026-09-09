import pandas as pd
import pytest

from utils.csv_utils import completed_classes


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
