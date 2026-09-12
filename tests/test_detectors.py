"""Factory detector va cac phep chuyen doi kem theo.

Nap SAA/detectors.py theo duong dan, bo qua SAA/__init__.py (no import .model
va keo theo torch). Chi kha thi khi detectors.py khong co import nang o cap
module - dung rang buoc thiet ke cua file do.
"""

import importlib.util
import os

import pytest

_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'SAA', 'detectors.py'
)


def _load():
    spec = importlib.util.spec_from_file_location('saa_detectors', _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


detectors = _load()


def test_module_has_no_heavy_top_level_imports():
    assert detectors.DETECTORS


def test_detector_names():
    assert set(detectors.DETECTORS) == {'grounding_dino', 'yolo_world', 'owlv2'}


def test_query_scored_excludes_grounding_dino():
    """grounding_dino di duong cu (ma tran diem theo token), khong phai query."""
    assert detectors.QUERY_SCORED == {'yolo_world', 'owlv2'}


def test_unknown_detector_lists_the_valid_names():
    with pytest.raises(ValueError) as excinfo:
        detectors.build_detector('detr', device='cpu')

    message = str(excinfo.value)
    assert 'detr' in message
    assert 'grounding_dino' in message


def test_build_detector_refuses_grounding_dino():
    """DINO duoc dung o SAA/model.py qua load_dino, khong qua factory nay."""
    with pytest.raises(ValueError) as excinfo:
        detectors.build_detector('grounding_dino', device='cpu')

    assert 'load_dino' in str(excinfo.value)


@pytest.mark.parametrize('phrase, expected', [
    ('blue defect. black defect. scratch.', ['blue defect', 'black defect', 'scratch']),
    ('black hole', ['black hole']),
    ('defect.', ['defect']),
    ('  thread .  hole  ', ['thread', 'hole']),
    ('carpet,', ['carpet,']),          # quirk dau phay o spec muc 2.4, giu nguyen
])
def test_split_phrase(phrase, expected):
    assert detectors.split_phrase(phrase) == expected


def test_split_phrase_drops_empty_terms():
    assert detectors.split_phrase('a.. b.') == ['a', 'b']


def test_xyxy_to_cxcywh_norm():
    """Sai he toa do la loi im lang: dien tich sai -> defect_max_area sai ->
    anomaly map van ra so nhin co ve hop ly."""
    boxes = [(10.0, 20.0, 30.0, 60.0)]      # xyxy pixel

    out = detectors.xyxy_to_cxcywh_norm(boxes, width=100, height=200)

    assert out == [pytest.approx((0.2, 0.2, 0.2, 0.2))]


def test_xyxy_to_cxcywh_norm_full_image():
    out = detectors.xyxy_to_cxcywh_norm([(0.0, 0.0, 100.0, 200.0)], width=100, height=200)

    assert out == [pytest.approx((0.5, 0.5, 1.0, 1.0))]


def test_xyxy_to_cxcywh_norm_empty():
    assert detectors.xyxy_to_cxcywh_norm([], width=100, height=200) == []
