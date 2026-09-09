import importlib.util
import os

import pytest

_BACKBONES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'SAA',
    'backbones.py',
)


def _load_backbones():
    """Nap SAA/backbones.py theo duong dan, bo qua SAA/__init__.py.

    SAA/__init__.py import .model va keo theo torch. Nap truc tiep giup
    test bang ten va duong loi chay duoc tren may khong co torch.
    """
    spec = importlib.util.spec_from_file_location('saa_backbones', _BACKBONES_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backbones = _load_backbones()


def test_module_has_no_heavy_top_level_imports():
    """Neu file nay load duoc tren venv khong co torch thi rang buoc con giu."""
    assert backbones.SAM_VARIANTS


def test_sam_variant_names_cover_the_config_grid():
    """Ba cau hinh Lite o muc 4.2 cua spec, cong baseline."""
    assert set(backbones.SAM_VARIANTS) == {'vit_h', 'mobile_sam', 'efficientvit_l0'}


def test_saliency_backbone_names_cover_the_config_grid():
    assert set(backbones.SALIENCY_BACKBONES) == {'wide_resnet50', 'mobilenetv3'}


def test_wide_resnet50_maps_to_the_timm_name_used_by_the_baseline():
    assert backbones.SALIENCY_BACKBONES['wide_resnet50'] == 'wide_resnet50_2'


def test_unknown_sam_variant_lists_the_valid_names():
    with pytest.raises(ValueError) as excinfo:
        backbones.build_sam_predictor('vit_gigantic', checkpoint='x.pth', device='cpu')

    message = str(excinfo.value)
    assert 'vit_gigantic' in message
    assert 'vit_h' in message


def test_unknown_saliency_backbone_lists_the_valid_names():
    with pytest.raises(ValueError) as excinfo:
        backbones.build_saliency_extractor('resnet9000', device='cpu')

    message = str(excinfo.value)
    assert 'resnet9000' in message
    assert 'wide_resnet50' in message
