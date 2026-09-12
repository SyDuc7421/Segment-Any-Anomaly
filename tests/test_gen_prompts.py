"""Sinh prompt spec bang VLM cuc bo.

Test khong nap model - chi kiem ranh gioi ro ri du lieu, dung khoi message, va
viec boc JSON ra khoi output. Nap module theo duong dan de tranh SAA/__init__.py
keo theo torch.
"""

import importlib.util
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        'gen_prompts', os.path.join(_ROOT, 'tools', 'gen_prompts.py')
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gp = _load()


@pytest.fixture
def fake_dataset(tmp_path):
    """Cay thu muc giong MVTec, co ca anh test va ground truth."""
    root = tmp_path / 'mvtec'
    for sub in ('train/good', 'test/good', 'test/color', 'ground_truth/color'):
        (root / 'carpet' / sub).mkdir(parents=True)
        for i in range(3):
            (root / 'carpet' / sub / f'{i:03d}.png').write_bytes(b'x')
    return str(root)


def test_only_train_good_images_are_returned(fake_dataset):
    """Ranh gioi chong ro ri du lieu, spec muc 5.5."""
    paths = gp.train_image_paths('mvtec', 'carpet', fake_dataset, limit=10)

    assert paths, 'phai tra ve it nhat mot anh'
    for p in paths:
        assert os.sep + 'train' + os.sep in p, f'anh ngoai train split: {p}'
        assert os.sep + 'test' + os.sep not in p
        assert 'ground_truth' not in p


def test_limit_is_respected(fake_dataset):
    assert len(gp.train_image_paths('mvtec', 'carpet', fake_dataset, limit=2)) == 2


def test_paths_are_sorted_for_reproducibility(fake_dataset):
    """Greedy decoding chi tai lap duoc neu dau vao co thu tu on dinh."""
    paths = gp.train_image_paths('mvtec', 'carpet', fake_dataset, limit=10)

    assert paths == sorted(paths)


def test_missing_class_directory_returns_empty(fake_dataset):
    assert gp.train_image_paths('mvtec', 'khong_ton_tai', fake_dataset, limit=3) == []


def test_blind_variant_has_no_image_block():
    content = gp.build_messages('carpet', n_images=0)[-1]['content']

    assert all(block['type'] == 'text' for block in content)


def test_vision_variant_puts_images_before_text():
    content = gp.build_messages('carpet', n_images=2)[-1]['content']

    assert [b['type'] for b in content] == ['image', 'image', 'text']


def test_class_name_reaches_the_prompt():
    content = gp.build_messages('metal_nut', n_images=0)[-1]['content']

    assert 'metal_nut' in content[-1]['text']


def test_system_prompt_states_the_cost_of_each_prompt():
    """LLM phai biet moi prompt ton mot luot detector, neu khong no se liet ke
    cang nhieu cang tot va tu bo mat truc toc do."""
    assert 'forward pass' in gp.SYSTEM


def test_system_prompt_does_not_leak_test_set_findings():
    """Thang so sanh do TREN TAP TEST rang texture hop voi prompt generic hon.
    Nhet phat hien do vao prompt he thong la ma hoa ket qua tap test vao bo sinh
    prompt - P2 mat tu cach zero-shot ngay."""
    lowered = (gp.SYSTEM + gp.USER_TEMPLATE).lower()

    for banned in ('texture', 'generic prompts work', 'fewer prompts score'):
        assert banned not in lowered, f'ro ri phat hien tu tap test: {banned!r}'


def test_parse_spec_extracts_json_from_prose():
    """Model trong so mo hay boc JSON trong ```json ... ``` hoac them loi dan."""
    text = ('Here is the spec:\n```json\n{"object_prompt": "carpet", '
            '"object_number": 1, "k_mask": 5, "defect_area_threshold": 0.9, '
            '"defect_prompts": [{"text": "hole", "filter": "carpet"}]}\n```\nHope that helps!')

    spec = gp.parse_spec(text, 'carpet')

    assert spec['class'] == 'carpet'
    assert spec['object_prompt'] == 'carpet'


def test_parse_spec_accepts_bare_json():
    text = ('{"object_prompt": "grid", "object_number": 1, "k_mask": 5, '
            '"defect_area_threshold": 0.9, '
            '"defect_prompts": [{"text": "bent wire", "filter": "grid"}]}')

    assert gp.parse_spec(text, 'grid')['object_prompt'] == 'grid'


def test_parse_spec_rejects_output_with_no_json():
    with pytest.raises(ValueError):
        gp.parse_spec('I am not sure what you want.', 'carpet')


def test_parse_spec_rejects_broken_json():
    with pytest.raises(ValueError):
        gp.parse_spec('{"object_prompt": "carpet", oops}', 'carpet')


def test_parse_spec_validates_the_schema():
    """Thieu truong phai lo ra o day, khong phai giua lan chay GPU."""
    with pytest.raises(ValueError):
        gp.parse_spec('{"object_prompt": "carpet", "defect_prompts": []}', 'carpet')
