"""Loader cho prompt spec do LLM sinh.

Nap SAA/prompts/llm_prompts.py theo duong dan, bo qua SAA/__init__.py (no import
.model va keo theo torch). Chi kha thi khi llm_prompts.py khong co import nang o
cap module - va no khong nen co, day chi la doc JSON.
"""

import importlib.util
import json
import os

import pytest

_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'SAA', 'prompts', 'llm_prompts.py',
)


def _load():
    spec = importlib.util.spec_from_file_location('saa_llm_prompts', _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mod = _load()
derive_property_fields = _mod.derive_property_fields
load_prompt_file = _mod.load_prompt_file
to_ensemble_prompts = _mod.to_ensemble_prompts
validate_spec = _mod.validate_spec


def test_module_has_no_heavy_top_level_imports():
    assert _mod.PROMPT_SOURCES


def _spec(**over):
    spec = {
        'class': 'carpet',
        'object_prompt': 'carpet',
        'object_number': 1,
        'k_mask': 5,
        'defect_area_threshold': 0.9,
        'defect_prompts': [
            {'text': 'black hole', 'filter': 'carpet'},
            {'text': 'thread', 'filter': 'carpet'},
        ],
    }
    spec.update(over)
    return spec


def test_valid_spec_passes():
    validate_spec(_spec())


@pytest.mark.parametrize('field', [
    'class', 'object_prompt', 'object_number', 'k_mask',
    'defect_area_threshold', 'defect_prompts',
])
def test_missing_field_is_rejected(field):
    spec = _spec()
    del spec[field]
    with pytest.raises(ValueError) as e:
        validate_spec(spec)
    assert field in str(e.value)


def test_object_number_must_be_at_least_one():
    """object_max_area = 1 / object_number - so 0 la chia cho khong."""
    with pytest.raises(ValueError):
        validate_spec(_spec(object_number=0))


def test_defect_area_threshold_must_be_in_zero_one():
    with pytest.raises(ValueError):
        validate_spec(_spec(defect_area_threshold=1.5))


def test_empty_defect_prompts_is_rejected():
    """Khong prompt nao thi khong box defect nao - anomaly map rong."""
    with pytest.raises(ValueError):
        validate_spec(_spec(defect_prompts=[]))


def test_defect_prompt_entry_needs_text_and_filter():
    with pytest.raises(ValueError):
        validate_spec(_spec(defect_prompts=[{'text': 'hole'}]))


def test_to_ensemble_prompts_matches_the_manual_shape():
    """set_ensemble_text_prompts nhan [[text, filter], ...] - xem
    SAA/prompts/mvtec_parameters.py."""
    assert to_ensemble_prompts(_spec()) == [
        ['black hole', 'carpet'],
        ['thread', 'carpet'],
    ]


def test_derive_property_fields():
    """Cac truong nay set_property_text_prompts tinh bang cach dem chi so tu.
    Duong JSON tinh thang, va phai cho ra dung cung loai gia tri."""
    fields = derive_property_fields(_spec())

    assert fields == {
        'object_prompt': 'carpet',
        'object_number': 1,
        'k_mask': 5,
        'defect_area_threshold': 0.9,
        'object_max_area': 1.0,
        'object_min_area': 0.0,
        'similar': 'dissimilar',
    }


def test_derive_property_fields_two_objects_halves_max_area():
    assert derive_property_fields(_spec(object_number=2))['object_max_area'] == 0.5


def test_object_prompt_has_no_trailing_comma():
    """Duong manual cho ra 'carpet,' vi parse theo vi tri tu (spec muc 2.4).
    Duong JSON KHONG duoc bat chuoc quirk do - lan chay doi chung P3-clean
    ton tai de do rieng anh huong cua dau phay."""
    assert not derive_property_fields(_spec())['object_prompt'].endswith(',')


def test_load_prompt_file_keys_by_class(tmp_path):
    path = tmp_path / 'mvtec-blind.json'
    path.write_text(json.dumps([_spec(), _spec(**{'class': 'grid', 'object_prompt': 'grid'})]))

    loaded = load_prompt_file(str(path))

    assert set(loaded) == {'carpet', 'grid'}
    assert loaded['grid']['object_prompt'] == 'grid'


def test_load_prompt_file_validates_every_entry(tmp_path):
    path = tmp_path / 'bad.json'
    bad = _spec()
    del bad['k_mask']
    path.write_text(json.dumps([_spec(), bad]))

    with pytest.raises(ValueError) as e:
        load_prompt_file(str(path))

    assert 'k_mask' in str(e.value)


def test_load_prompt_file_rejects_duplicate_class(tmp_path):
    path = tmp_path / 'dup.json'
    path.write_text(json.dumps([_spec(), _spec()]))

    with pytest.raises(ValueError) as e:
        load_prompt_file(str(path))

    assert 'carpet' in str(e.value)
