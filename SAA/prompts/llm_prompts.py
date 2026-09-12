"""Nap prompt do LLM sinh tu JSON.

Duong nay ton tai de tranh hoan toan tro parse theo vi tri tu o
set_property_text_prompts (spec muc 2.4): `property_prompts.split(' ')[7]` qua
mong manh de LLM sinh ra an toan. Nhanh manual giu nguyen, khong sua mot dong.
"""

import json

PROMPT_SOURCES = ('manual', 'general', 'generic', 'llm')

_REQUIRED = ('class', 'object_prompt', 'object_number', 'k_mask',
             'defect_area_threshold', 'defect_prompts')


def validate_spec(spec):
    """Raise ValueError neu spec khong dung schema o spec muc 5.2."""
    for field in _REQUIRED:
        if field not in spec:
            raise ValueError(f"thieu truong '{field}' trong spec: {spec.get('class', '?')}")

    if not isinstance(spec['object_number'], int) or spec['object_number'] < 1:
        raise ValueError(
            f"object_number phai la so nguyen >= 1 (object_max_area = 1 / object_number), "
            f"nhan duoc {spec['object_number']!r}"
        )

    if not isinstance(spec['k_mask'], int) or spec['k_mask'] < 1:
        raise ValueError(f"k_mask phai la so nguyen >= 1, nhan duoc {spec['k_mask']!r}")

    threshold = spec['defect_area_threshold']
    if not 0 < threshold <= 1:
        raise ValueError(f"defect_area_threshold phai trong (0, 1], nhan duoc {threshold!r}")

    if not spec['defect_prompts']:
        raise ValueError(
            f"defect_prompts rong cho class {spec['class']}: khong prompt nao thi "
            f"khong box defect nao, anomaly map se rong"
        )

    for entry in spec['defect_prompts']:
        missing = {'text', 'filter'} - set(entry)
        if missing:
            raise ValueError(f"defect_prompt thieu {sorted(missing)}: {entry}")


def to_ensemble_prompts(spec):
    """Shape ma set_ensemble_text_prompts nhan: [[text, filter], ...]."""
    return [[entry['text'], entry['filter']] for entry in spec['defect_prompts']]


def derive_property_fields(spec):
    """Cac truong ma set_property_text_prompts tinh bang cach dem chi so tu.

    `similar` khoa cung la 'dissimilar': moi cau trong SAA/prompts/*_parameters.py
    deu dung tu do o vi tri [6], va no khong phai tham so ma LLM nen quyet dinh.
    """
    return {
        'object_prompt': spec['object_prompt'],
        'object_number': spec['object_number'],
        'k_mask': spec['k_mask'],
        'defect_area_threshold': spec['defect_area_threshold'],
        'object_max_area': 1. / spec['object_number'],
        'object_min_area': 0.,
        'similar': 'dissimilar',
    }


def load_prompt_file(path):
    """Doc file JSON, tra ve dict khoa theo ten class. Validate tung entry."""
    with open(path) as f:
        specs = json.load(f)

    loaded = {}
    for spec in specs:
        validate_spec(spec)
        name = spec['class']
        if name in loaded:
            raise ValueError(f"class '{name}' xuat hien hai lan trong {path}")
        loaded[name] = spec

    return loaded
