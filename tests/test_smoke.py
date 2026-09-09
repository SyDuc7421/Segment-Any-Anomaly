def test_repo_root_is_importable():
    import utils.csv_utils

    assert hasattr(utils.csv_utils, 'write_results')
