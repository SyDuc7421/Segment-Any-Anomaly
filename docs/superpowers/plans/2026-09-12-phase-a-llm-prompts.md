# Phase A — LLM sinh prompt: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thay tu dien prompt viet tay theo tung class bang prompt do LLM sinh
ra, do khoang cach toi prompt cua tac gia goc, va do luon anh huong len toc do —
vi so luot goi Grounding DINO ty le thuan voi so prompt.

**Architecture:** Nhanh song song, khong refactor. Duong `manual` hien tai giu
nguyen tung dong. Prompt do LLM sinh di qua mot loader JSON moi va mot setter
moi (`set_property_from_dict`), tranh hoan toan tro parse theo vi tri tu o spec
muc 2.4. Mot co CLI `--prompt-source` chon nhanh.

**Tech Stack:** Python, OpenAI SDK (`openai`), model `gpt-5.6-luna`, Pydantic, pytest.
Khoa API doc tu `OPENAI_API_KEY` trong `.env` (da gitignore o dong 107, chua track).

**Spec:** `docs/superpowers/specs/2026-08-28-saa-lite-llm-prompt-design.md` muc 5

**Bao cao Phase B:** `docs/report-phase-b.md` — doc muc 6b truoc, vi ket qua Phase
B doi ban chat cua Phase A.

## Vi sao Phase A gio co HAI muc tieu

Spec ban dau chi dat mot muc tieu: chung minh LLM thay duoc chuyen gia. Phase B
them muc tieu thu hai, va no khong phai phat sinh — no la thu con lai sau khi
loai tru bang thuc nghiem moi duong khac.

| Don bay toc do | Ket qua do duoc |
|---|---|
| Thay SAM (ViT-H -> MobileSAM) | 2.07x, da vat 90% du dia truc SAM |
| Thay saliency backbone | 1% thoi gian, khong dang |
| Thay detector (YOLO-World, OWLv2) | **That bai** — vua kem chinh xac hon vua cham hon |
| **Giam so luot goi DINO** | **Chua do — duong duy nhat con lai** |

DINO chiem 72% thoi gian tren MVTec va khong thay the duoc. So luot goi moi anh
= `1 object + 3 general + K manual`. Class `carpet` co K=3 -> 7 luot, moi luot
~267 ms.

Neu LLM sinh 3 prompt trung thay vi 6:

```
4 luot x 267 = 1067 ms
t_total ~ 1067 + 99 + 25 = 1191 ms
speedup so baseline = 3.57x     <- vuot moc >= 3x ma Phase B khong voi toi
```

Day la uoc tinh, chua do. Nhung no la ly do Phase A phai **dem va bao cao so
luot DINO**, khong chi bao cao accuracy.

## Global Constraints

- **Tham so khoa cung** (spec muc 6.1): `eval_resolution=400`, `box_threshold=0.1`,
  `text_threshold=0.1`, `experiment_indx=0` (seed 111). Doi la chay lai tat ca.
- **Cau hinh chay**: lite2 (`sam_variant=mobile_sam`, `saliency_backbone=mobilenetv3`,
  `detector=grounding_dino`) — thang Buoc 1, xac nhan o Buoc 2.
- **Duong `manual` bit-exact**: `set_property_text_prompts()` va
  `SAA/prompts/*_parameters.py` **khong sua mot dong nao**. P3 chinh la lan chay
  o `results/lite2_full/`, khong chay lai.
- **Ranh gioi ro ri du lieu** (spec muc 5.5): script sinh prompt **chi doc `train`
  split**, hard-code trong code. Khong cham anh test, khong cham ground-truth,
  khong cham anh anomaly. De lot la toan bo ket qua mat gia tri.
- **JSON sinh ra phai commit vao git**: dieu kien de hoi dong tai lap ma khong
  can API key.
- **Full test set**: khong dung `--max-samples` cho bat ky con so nao vao luan van.
- **LLM**: `gpt-5.6-luna` qua OpenAI SDK. Khoa doc tu `.env`, **khong hard-code,
  khong in ra log**. `.env` da nam trong `.gitignore`; kiem lai truoc khi commit.
- **Bo self-refine** (spec muc 5.7): MVTec khong co validation split, moi subset
  de cham deu phai cat tu test, tu ro ri.

## Mot confound phai xu ly, khong duoc lam ngo

`set_property_text_prompts` parse theo vi tri tu:

```python
self.object_prompt = property_prompts.split(' ')[7]
```

Voi cau mau `'the image of carpet have 1 dissimilar carpet, with a maximum of 5
anomaly...'`, chi so `[7]` cho ra `'carpet,'` — **dinh dau phay**. Chuoi do di
thang vao DINO lam prompt do object.

Duong JSON se cho ra `'carpet'` sach. Nghia la P2 va P3 khac nhau **hai bien**
cung luc: noi dung prompt, va dau phay. Khong tach duoc thi khong ket luan duoc.

**Xu ly**: them mot lan chay doi chung **P3-clean** — dung dung prompt thu cong
cua tac gia nhung nap qua duong JSON (khong dau phay). So P3 voi P3-clean co lap
duoc anh huong cua dau phay; so P3-clean voi P2 moi la so noi dung prompt.

Ton them mot lan chay MVTec (~0.9h). Dang, vi khong co no thi moi so sanh P2-P3
deu co the bi van lai.

---

## File Structure

| File | Trang thai | Trach nhiem |
|---|---|---|
| `SAA/prompts/llm_prompts.py` | Moi | Doc JSON, tra ve dung shape ma `eval_SAA.py` dang dung |
| `SAA/prompts/generated/mvtec-blind.json` | Moi | Output LLM, **commit** |
| `SAA/prompts/generated/mvtec-vision.json` | Moi | " |
| `SAA/prompts/generated/visa_public-blind.json` | Moi | " |
| `SAA/prompts/generated/visa_public-vision.json` | Moi | " |
| `tools/gen_prompts.py` | Moi | Goi OpenAI API (`gpt-5.6-luna`) sinh JSON |
| `SAA/model.py` | Sua | Them `set_property_from_dict()` |
| `eval_SAA.py` | Sua | Them `--prompt-source`, `--llm-prompt-file` |
| `run_MVTec.py`, `run_VisA_public.py` | Sua | Doc `PROMPT_SOURCE`, `LLM_PROMPT_FILE`; dua vao run_identity |
| `tests/test_llm_prompts.py` | Moi | Schema, loader, ranh gioi ro ri |
| `demo/Phase_A_LLM_Prompts.ipynb` | Moi | Sinh prompt + chay 5 muc |

Khong dung toi `utils/metrics.py`, `utils/timing.py`, `SAA/backbones.py`,
`SAA/detectors.py`.

---

## Task 1: Schema va loader

**Files:**
- Create: `SAA/prompts/llm_prompts.py`
- Test: `tests/test_llm_prompts.py`

**Interfaces:**
- Produces:
  - `PROMPT_SOURCES = {'manual', 'general', 'generic', 'llm'}`
  - `validate_spec(spec: dict) -> None` — raise `ValueError` neu sai schema
  - `load_prompt_file(path) -> dict[str, dict]` — khoa la ten class
  - `to_ensemble_prompts(spec) -> list[list[str]]` — shape `[[text, filter], ...]` ma `set_ensemble_text_prompts` dang nhan
  - `derive_property_fields(spec) -> dict` — cac truong ma `set_property_from_dict` se gan

Schema (spec muc 5.2):

```json
{
  "class": "carpet",
  "object_prompt": "carpet",
  "object_number": 1,
  "k_mask": 5,
  "defect_area_threshold": 0.9,
  "defect_prompts": [
    {"text": "black hole", "filter": "carpet"},
    {"text": "thread", "filter": "carpet"}
  ]
}
```

- [ ] **Step 1: Viet test that bai**

```python
import json

import pytest

from SAA.prompts.llm_prompts import (
    derive_property_fields,
    load_prompt_file,
    to_ensemble_prompts,
    validate_spec,
)


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
    fields = derive_property_fields(_spec(object_number=2))

    assert fields['object_max_area'] == 0.5


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
```

- [ ] **Step 2: Chay test, xac nhan FAIL**

Run: `.venv-test/bin/pytest tests/test_llm_prompts.py -v`
Expected: FAIL voi `ModuleNotFoundError: No module named 'SAA.prompts.llm_prompts'`.

Neu FAIL vi `SAA/__init__.py` keo theo torch, nap module theo duong dan nhu
`tests/test_detectors.py` dang lam.

- [ ] **Step 3: Viet `SAA/prompts/llm_prompts.py`**

```python
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

    `similar` khoa cung la 'dissimilar': moi cau trong
    SAA/prompts/*_parameters.py deu dung tu do o vi tri [6], va no khong phai
    tham so ma LLM nen quyet dinh.
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
```

- [ ] **Step 4: Chay test, xac nhan PASS**

Run: `.venv-test/bin/pytest tests/test_llm_prompts.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add SAA/prompts/llm_prompts.py tests/test_llm_prompts.py
git commit -m "feat: load LLM-generated prompt specs from JSON"
```

---

## Task 2: `set_property_from_dict` trong `SAA/model.py`

**Files:**
- Modify: `SAA/model.py`

**Interfaces:**
- Consumes: `derive_property_fields` tu Task 1.
- Produces: `Model.set_property_from_dict(spec, verbose=False)` gan dung nhung
  thuoc tinh ma `set_property_text_prompts` gan.

- [ ] **Step 1: Them method, ngay SAU `set_property_text_prompts`**

**Khong sua `set_property_text_prompts`.** Method moi dat ngay duoi no.

```python
    def set_property_from_dict(self, spec, verbose=False):
        """Gan property prompt tu spec JSON thay vi tu chuoi.

        Tuong duong set_property_text_prompts nhung khong dem chi so tu. Xem
        spec muc 2.4: `property_prompts.split(' ')[7]` cho ra 'carpet,' dinh
        dau phay - hanh vi baseline giu nguyen o nhanh manual, con nhanh nay
        cho ra chuoi sach.
        """
        from .prompts.llm_prompts import derive_property_fields

        fields = derive_property_fields(spec)
        for name, value in fields.items():
            setattr(self, name, value)

        if verbose:
            print('used property prompts (from dict) ===')
            for name, value in fields.items():
                print(f'{name}: {value}')
            print('=====================================')
```

- [ ] **Step 2: Kiem tra khong lech thuoc tinh nao**

`set_property_text_prompts` gan bay thuoc tinh. Doi chieu:

```bash
grep -n "self\." SAA/model.py | sed -n '/def set_property_text_prompts/,/def /p'
```

Chay:
```bash
.venv-test/bin/python -c "
import ast, sys
src = open('SAA/model.py').read()
tree = ast.parse(src)
cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'Model')
def assigned(fn_name):
    fn = next(f for f in cls.body if getattr(f, 'name', None) == fn_name)
    return {t.attr for n in ast.walk(fn) for t in ast.walk(n)
            if isinstance(t, ast.Attribute) and isinstance(t.ctx, ast.Store)}
a, b = assigned('set_property_text_prompts'), assigned('set_property_from_dict')
print('manual:', sorted(a)); print('dict  :', sorted(b))
print('THIEU:', sorted(a - b) or 'khong')
"
```

Expected: `THIEU: khong`. Con thuoc tinh nao thieu la pipeline se doc gia tri cu
tu lan chay truoc — mot loi im lang.

- [ ] **Step 3: py_compile va chay lai suite**

Run: `.venv-test/bin/python -m py_compile SAA/model.py && .venv-test/bin/pytest tests/ -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add SAA/model.py
git commit -m "feat: set property prompts from a dict, bypassing word-position parsing"
```

---

## Task 3: `--prompt-source` trong `eval_SAA.py` va runner

**Files:**
- Modify: `eval_SAA.py`
- Modify: `run_MVTec.py`, `run_VisA_public.py`

**Interfaces:**
- Consumes: `load_prompt_file`, `to_ensemble_prompts` (Task 1);
  `set_property_from_dict` (Task 2).
- Produces: `--prompt-source {manual,general,generic,llm}`, `--llm-prompt-file PATH`;
  bien moi truong `PROMPT_SOURCE`, `LLM_PROMPT_FILE`.

- [ ] **Step 1: Thay khoi ghep prompt trong `main()`**

Khoi hien tai:

```python
    general_prompts = SegmentAnyAnomaly.build_general_prompts(kwargs['class_name'])
    manual_promts = SegmentAnyAnomaly.manul_prompts[kwargs['dataset']][kwargs['class_name']]

    textual_prompts = general_prompts + manual_promts

    model.set_ensemble_text_prompts(textual_prompts, verbose=False)

    property_text_prompts = SegmentAnyAnomaly.property_prompts[kwargs['dataset']][kwargs['class_name']]
    model.set_property_text_prompts(property_text_prompts, verbose=False)
```

thanh:

```python
    # Thang so sanh o spec muc 5.6. Nhanh 'manual' la P3 va phai giu nguyen
    # tung dong - moi con so baseline deu tu no.
    prompt_source = kwargs['prompt_source']
    class_name = kwargs['class_name']
    dataset = kwargs['dataset']

    general_prompts = SegmentAnyAnomaly.build_general_prompts(class_name)

    if prompt_source == 'manual':
        manual_promts = SegmentAnyAnomaly.manul_prompts[dataset][class_name]
        textual_prompts = general_prompts + manual_promts
    elif prompt_source == 'general':
        textual_prompts = general_prompts          # P1: san co san
    elif prompt_source == 'generic':
        textual_prompts = [['defect.', class_name]]  # P0: san tuyet doi
    else:
        from SAA.prompts.llm_prompts import load_prompt_file, to_ensemble_prompts
        specs = load_prompt_file(kwargs['llm_prompt_file'])
        if class_name not in specs:
            raise ValueError(
                f"{kwargs['llm_prompt_file']} khong co class '{class_name}'. "
                f"Co: {sorted(specs)}"
            )
        llm_spec = specs[class_name]
        textual_prompts = to_ensemble_prompts(llm_spec)

    logger.info(f'prompt_source={prompt_source}, {len(textual_prompts)} prompt: '
                f'{[p[0] for p in textual_prompts]}')

    model.set_ensemble_text_prompts(textual_prompts, verbose=False)

    if prompt_source == 'llm':
        model.set_property_from_dict(llm_spec, verbose=False)
    else:
        property_text_prompts = SegmentAnyAnomaly.property_prompts[dataset][class_name]
        model.set_property_text_prompts(property_text_prompts, verbose=False)
```

Luu y: P0 va P1 **khong** dung `general_prompts + manual`, nen so luot DINO giam
theo. Dong log in ra so prompt chinh la du lieu cho muc tieu 2.

- [ ] **Step 2: Them hai argument**

```python
    parser.add_argument('--prompt-source', type=str, default='manual',
                        choices=['manual', 'general', 'generic', 'llm'],
                        help='manual=P3 (baseline), general=P1, generic=P0, llm=P2')
    parser.add_argument('--llm-prompt-file', type=str, default=None,
                        help='Duong dan JSON khi --prompt-source llm')
```

Va them ca hai vao `run_meta.json`:

```python
        'prompt_source': kwargs['prompt_source'],
        'llm_prompt_file': kwargs['llm_prompt_file'],
```

- [ ] **Step 3: Runner doc bien moi truong**

Trong ca hai runner, canh `detector`:

```python
    prompt_source = os.environ.get('PROMPT_SOURCE') or 'manual'
    llm_prompt_file = os.environ.get('LLM_PROMPT_FILE') or None
```

Dua **ca hai** vao `run_identity` — doi nguon prompt la doi con so:

```python
        'prompt_source': prompt_source,
        'llm_prompt_file': llm_prompt_file,
```

Va vao lenh:

```python
                '--prompt-source', run_identity['prompt_source'],
```
```python
            if run_identity['llm_prompt_file'] is not None:
                cmd += ['--llm-prompt-file', run_identity['llm_prompt_file']]
```

- [ ] **Step 4: Xac nhan nhanh manual khong doi**

```bash
.venv-test/bin/python -m py_compile eval_SAA.py run_MVTec.py run_VisA_public.py
git diff -w eval_SAA.py
```

Doc ky: khong dong nao trong nhanh `manual` duoc doi bieu thuc. Chi la thut le
vao trong `if`.

- [ ] **Step 5: Chay lai suite va commit**

```bash
.venv-test/bin/pytest tests/ -q
git add eval_SAA.py run_MVTec.py run_VisA_public.py
git commit -m "feat: select the prompt source, from generic floor to LLM to manual"
```

---

## Task 4: `tools/gen_prompts.py`

**Files:**
- Create: `tools/gen_prompts.py`, `tools/__init__.py`
- Test: `tests/test_gen_prompts.py`

**Interfaces:**
- Produces:
  - `train_image_paths(dataset, class_name, root, limit) -> list[str]` — **chi tra ve anh trong `train/good`**
  - `build_user_content(class_name, image_paths) -> list[dict]`
  - `PromptSpec` (Pydantic) — schema ma API tra ve
  - CLI: `python tools/gen_prompts.py --dataset mvtec --variant blind --out SAA/prompts/generated/mvtec-blind.json`

**Ranh gioi ro ri la phan quan trong nhat cua task nay.** Spec muc 5.5: script
chi duoc doc `train` split, hard-code trong code. Luan van phai trich duoc dong
code chung minh. Test phai chan duoc moi duong khac.

### Hai cho chua chac chan, phai kiem truoc khi ton tien

Code duoi day viet theo Responses API cua OpenAI SDK. Hai thu **chua duoc kiem
chung voi `gpt-5.6-luna`**:

1. **Ten method structured output.** Ban dung `client.responses.parse(...,
   text_format=PromptSpec)`. Mot so ban SDK dung
   `client.beta.chat.completions.parse(..., response_format=...)` voi shape
   message khac han.
2. **Shape khoi anh.** Responses API dung
   `{'type': 'input_image', 'image_url': 'data:image/png;base64,...'}`;
   Chat Completions dung `{'type': 'image_url', 'image_url': {'url': ...}}`.

Step 1 duoi day la mot lan goi **mot class duy nhat** de kiem ca hai truoc khi
chay 54 lan. Sai thi sua theo tai lieu SDK dang cai, dung doan.

- [ ] **Step 1: Probe API bang mot lan goi**

Truoc khi viet gi them, xac nhan model va surface chay duoc:

```bash
pip install -q openai pydantic python-dotenv
python - <<'PY'
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()
assert os.environ.get('OPENAI_API_KEY'), 'OPENAI_API_KEY chua co trong .env'

class Probe(BaseModel):
    object_prompt: str
    defect_prompts: list[str]

client = OpenAI()
r = client.responses.parse(
    model='gpt-5.6-luna',
    input=[{'role': 'user',
            'content': [{'type': 'input_text',
                         'text': 'Industrial anomaly detection for carpet. '
                                 'Give the object noun and two defect phrases.'}]}],
    text_format=Probe,
)
print(r.output_parsed)
PY
```

Chay duoc thi di tiep. Loi `AttributeError` hoac `NotFoundError` tren ten model
thi ghi lai thong bao loi vao Notes va sua theo tai lieu cua ban SDK dang cai —
**dung suy tu shape cua SDK khac**.

Lam tuong tu voi mot anh de xac nhan shape `input_image` truoc khi chay variant
`vision`.

- [ ] **Step 2: Viet test that bai**

```python
import os

import pytest

from tools.gen_prompts import build_user_content, train_image_paths


@pytest.fixture
def fake_dataset(tmp_path):
    """Dung cay thu muc giong MVTec, co ca anh test va anh anomaly."""
    root = tmp_path / 'mvtec'
    for sub in ('train/good', 'test/good', 'test/color', 'ground_truth/color'):
        (root / 'carpet' / sub).mkdir(parents=True)
        for i in range(3):
            (root / 'carpet' / sub / f'{i:03d}.png').write_bytes(b'x')
    return str(root)


def test_only_train_good_images_are_returned(fake_dataset):
    paths = train_image_paths('mvtec', 'carpet', fake_dataset, limit=10)

    assert paths, 'phai tra ve it nhat mot anh'
    for p in paths:
        assert os.sep + 'train' + os.sep in p, f'anh ngoai train split: {p}'
        assert os.sep + 'test' + os.sep not in p
        assert 'ground_truth' not in p


def test_limit_is_respected(fake_dataset):
    assert len(train_image_paths('mvtec', 'carpet', fake_dataset, limit=2)) == 2


def test_paths_are_sorted_for_reproducibility(fake_dataset):
    paths = train_image_paths('mvtec', 'carpet', fake_dataset, limit=10)

    assert paths == sorted(paths)


def test_blind_variant_sends_no_image():
    content = build_user_content('carpet', image_paths=[])

    assert all(block['type'] == 'input_text' for block in content)


def test_vision_variant_sends_images_before_text(fake_dataset):
    paths = train_image_paths('mvtec', 'carpet', fake_dataset, limit=2)

    content = build_user_content('carpet', image_paths=paths)

    assert [b['type'] for b in content] == ['input_image', 'input_image', 'input_text']
    assert content[0]['image_url'].startswith('data:image/png;base64,')


def test_class_name_reaches_the_prompt():
    content = build_user_content('metal_nut', image_paths=[])

    assert 'metal_nut' in content[-1]['text']
```

- [ ] **Step 3: Chay test, xac nhan FAIL**

Run: `.venv-test/bin/pytest tests/test_gen_prompts.py -v`
Expected: FAIL voi `ModuleNotFoundError: No module named 'tools'`.

- [ ] **Step 4: Viet `tools/gen_prompts.py`**

```python
"""Sinh prompt spec bang LLM, mot lan, roi cache vao git.

RANH GIOI RO RI DU LIEU (spec muc 5.5): script nay chi duoc doc `train` split.
Ham train_image_paths hard-code duong dan `train/good` va co test chan moi duong
khac. Khong cham anh test, khong cham ground-truth, khong cham anh anomaly. De
lot la toan bo ket qua Phase A mat gia tri.

Chay mot lan, output commit vao git - hoi dong tai lap duoc ma khong can API key.
"""

import argparse
import base64
import json
import os
from typing import List

from pydantic import BaseModel, Field

MODEL = 'gpt-5.6-luna'

SPLIT_DIRS = {
    # Chi train/good. Cay thu muc cua ca hai dataset deu theo dang nay.
    'mvtec': os.path.join('{class_name}', 'train', 'good'),
    'visa_public': os.path.join('{class_name}', 'train', 'good'),
}

SYSTEM = """You write text prompts for an open-vocabulary object detector \
(Grounding DINO) that will be used to find manufacturing defects.

The detector grounds short noun phrases in an image. Phrases that name a visible \
defect appearance work; abstract quality judgements do not. Prefer two-to-three \
word phrases naming what the defect looks like, not what caused it.

Every prompt you emit costs one detector forward pass per image, so a short list \
of precise phrases beats a long list of vague ones."""

USER_TEMPLATE = """Industrial anomaly detection, object category: {class_name}

Produce a prompt spec for this category:

- object_prompt: the noun the detector should use to find the object itself. \
Bare noun, no article, no punctuation.
- object_number: how many instances of that object appear in one image. Usually 1.
- k_mask: how many candidate defect regions to keep per image. 5 is a reasonable default.
- defect_area_threshold: the largest fraction of the object's area a single defect \
may occupy, in (0, 1]. 0.9 is a reasonable default.
- defect_prompts: the defect phrases. For each, `text` is the phrase given to the \
detector, and `filter` is a phrase that means the object itself - a box matching \
the filter is discarded as background, so it is normally the same as object_prompt.

Emit between two and five defect prompts. Fewer, more precise phrases are better \
than a long list."""


class DefectPrompt(BaseModel):
    text: str = Field(description="Phrase given to the detector")
    filter: str = Field(description="Phrase meaning the object itself; matching boxes are dropped")


class PromptSpec(BaseModel):
    object_prompt: str
    object_number: int
    k_mask: int
    defect_area_threshold: float
    defect_prompts: List[DefectPrompt]


def train_image_paths(dataset, class_name, root, limit):
    """Duong dan anh trong train/good, da sap xep.

    CHI train split. Day la ranh gioi chong ro ri du lieu o spec muc 5.5.
    """
    pattern = SPLIT_DIRS[dataset].format(class_name=class_name)
    directory = os.path.join(root, pattern)

    if not os.path.isdir(directory):
        return []

    names = sorted(n for n in os.listdir(directory)
                   if n.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')))

    return [os.path.join(directory, n) for n in names[:limit]]


def build_user_content(class_name, image_paths):
    """Noi dung message theo Responses API. Anh dat TRUOC text."""
    content = []

    for path in image_paths:
        with open(path, 'rb') as f:
            data = base64.standard_b64encode(f.read()).decode('utf-8')
        media_type = 'image/jpeg' if path.lower().endswith(('.jpg', '.jpeg')) else 'image/png'
        content.append({
            'type': 'input_image',
            'image_url': f'data:{media_type};base64,{data}',
        })

    content.append({
        'type': 'input_text',
        'text': USER_TEMPLATE.format(class_name=class_name),
    })
    return content


def generate_one(client, class_name, image_paths):
    response = client.responses.parse(
        model=MODEL,
        instructions=SYSTEM,
        input=[{'role': 'user', 'content': build_user_content(class_name, image_paths)}],
        text_format=PromptSpec,
    )

    spec = response.output_parsed.model_dump()
    spec['class'] = class_name
    return spec


def main():
    parser = argparse.ArgumentParser(description='Sinh prompt spec bang LLM')
    parser.add_argument('--dataset', choices=sorted(SPLIT_DIRS), required=True)
    parser.add_argument('--variant', choices=['blind', 'vision'], required=True)
    parser.add_argument('--data-root', required=True,
                        help='Thu muc goc dataset; chi train/good duoc doc')
    parser.add_argument('--out', required=True)
    parser.add_argument('--n-images', type=int, default=3,
                        help='So anh normal gui kem o variant vision')
    args = parser.parse_args()

    from dotenv import load_dotenv
    from openai import OpenAI

    from datasets import dataset_classes

    load_dotenv()
    if not os.environ.get('OPENAI_API_KEY'):
        raise SystemExit('OPENAI_API_KEY chua co - dat trong .env')

    client = OpenAI()
    specs = []

    for class_name in dataset_classes[args.dataset]:
        image_paths = []
        if args.variant == 'vision':
            image_paths = train_image_paths(
                args.dataset, class_name, args.data_root, args.n_images
            )
            if not image_paths:
                raise SystemExit(
                    f'khong tim thay anh train cho {class_name} trong {args.data_root} - '
                    f'variant vision can anh, dung im lang bo qua'
                )

        spec = generate_one(client, class_name, image_paths)
        print(f"{class_name}: {len(spec['defect_prompts'])} prompt  "
              f"{[p['text'] for p in spec['defect_prompts']]}")
        specs.append(spec)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(specs, f, indent=2, ensure_ascii=False)

    total = sum(len(s['defect_prompts']) for s in specs)
    print(f'\n{len(specs)} class, {total} prompt, trung binh '
          f'{total / len(specs):.1f} prompt/class -> {args.out}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 5: Chay test, xac nhan PASS**

Run: `.venv-test/bin/pytest tests/test_gen_prompts.py -v`
Expected: PASS. Can `.venv-test/bin/pip install pydantic openai python-dotenv`.

- [ ] **Step 6: Kiem tra ranh gioi ro ri bang grep**

```bash
grep -n "test\|ground_truth\|anomaly" tools/gen_prompts.py
```

Moi lan xuat hien phai la trong comment hoac trong ten bien khong dan toi doc
file. Co bat ky duong dan nao tro toi `test/` la **dung ngay**.

- [ ] **Step 7: Kiem tra khoa khong lot vao git**

```bash
git check-ignore -v .env          # phai in ra dong .gitignore khop
git ls-files --error-unmatch .env # phai bao "did not match any file"
grep -rn "sk-" tools/ SAA/prompts/generated/ 2>/dev/null || echo "khong co khoa trong ma nguon"
```

- [ ] **Step 8: Commit**

```bash
git add tools/ tests/test_gen_prompts.py
git commit -m "feat: generate prompt specs with an LLM, reading only the train split"
```

---

## Task 5: Sinh va commit JSON

**Files:**
- Create: `SAA/prompts/generated/*.json` (4 file)

Chay tren may co API key. Khong can GPU.

- [ ] **Step 1: Sinh bon file**

```bash
# OPENAI_API_KEY doc tu .env, khong can export

python tools/gen_prompts.py --dataset mvtec --variant blind \
    --data-root /path/to/mvtec --out SAA/prompts/generated/mvtec-blind.json
python tools/gen_prompts.py --dataset mvtec --variant vision \
    --data-root /path/to/mvtec --out SAA/prompts/generated/mvtec-vision.json
python tools/gen_prompts.py --dataset visa_public --variant blind \
    --data-root /path/to/visa --out SAA/prompts/generated/visa_public-blind.json
python tools/gen_prompts.py --dataset visa_public --variant vision \
    --data-root /path/to/visa --out SAA/prompts/generated/visa_public-vision.json
```

27 class x 2 variant = 54 loi goi. Vai do la.

- [ ] **Step 2: Ghi lai so prompt trung binh**

Script in ra o dong cuoi. Ghi vao Notes cua plan nay — **day la con so quyet dinh
muc tieu 2**. So luot DINO moi anh = `1 + so prompt`. Prompt thu cong cua tac gia
cho MVTec la `3 general + K manual`; neu LLM sinh it hon thi toc do tang theo.

- [ ] **Step 3: Kiem tra moi file nap duoc**

```bash
.venv-test/bin/python -c "
from SAA.prompts.llm_prompts import load_prompt_file
import glob
for p in sorted(glob.glob('SAA/prompts/generated/*.json')):
    d = load_prompt_file(p)
    n = sum(len(s['defect_prompts']) for s in d.values())
    print(f'{p}: {len(d)} class, {n} prompt, tb {n/len(d):.1f}')
"
```

`load_prompt_file` validate tung entry, nen loi schema lo ra o day chu khong
phai giua lan chay 1 tieng.

- [ ] **Step 4: Commit**

```bash
git add SAA/prompts/generated/
git commit -m "data: LLM-generated prompt specs for both datasets"
```

---

## Task 6: Notebook Phase A

**Files:**
- Create: `demo/Phase_A_LLM_Prompts.ipynb`

Dua tren `demo/Benchmark_SAA_Lite2.ipynb`: cung phan cai dat, cung MobileSAM,
cung cach ghi thang vao Drive. Khac o khoi cau hinh.

- [ ] **Step 1: Khoi chon muc**

```python
# Thang so sanh o spec muc 5.6. Doi LEVEL roi chay lai; moi muc mot thu muc Drive.
#
#   P0  generic     "defect."                      san tuyet doi
#   P1  general     general_prompts co san          san co san
#   P2b llm         mvtec-blind.json                LLM, chi biet ten class
#   P2v llm         mvtec-vision.json               LLM + anh normal
#   P3  manual      prompt thu cong                 tran - DA CHAY, xem run_lite2
#   P3c llm         manual-as-json.json             doi chung dau phay
LEVEL = 'P2b'

LEVELS = {
    'P0':  ('generic', None),
    'P1':  ('general', None),
    'P2b': ('llm', 'SAA/prompts/generated/{dataset}-blind.json'),
    'P2v': ('llm', 'SAA/prompts/generated/{dataset}-vision.json'),
    'P3c': ('llm', 'SAA/prompts/generated/{dataset}-manual-as-json.json'),
}

source, file_template = LEVELS[LEVEL]
os.environ['PROMPT_SOURCE'] = source
if file_template:
    os.environ['LLM_PROMPT_FILE'] = file_template.format(dataset='mvtec')
else:
    os.environ.pop('LLM_PROMPT_FILE', None)

# Cau hinh lite2, khoa cung - thang Buoc 1, xac nhan o Buoc 2
os.environ['SAM_VARIANT'] = 'mobile_sam'
os.environ['SALIENCY_BACKBONE'] = 'mobilenetv3'
os.environ['DETECTOR'] = 'grounding_dino'

ROOT_DIR = f'/content/drive/MyDrive/SAA_results/phase_a_{LEVEL}'
os.environ['ROOT_DIR'] = ROOT_DIR
os.makedirs(f'{ROOT_DIR}/csv', exist_ok=True)
```

`LLM_PROMPT_FILE` phai doi theo dataset, nen cell chay VisA dat lai bien truoc
khi goi `run_VisA_public.py`.

- [ ] **Step 2: Cell tong hop bac thang**

Doc moi thu muc `phase_a_*`, dat canh nhau, va tinh **khoang cach P1 -> P3 thu
hep duoc bao nhieu** — tieu chi spec muc 7 cho Phase A la >= 50%.

```python
import pandas as pd, glob, os

rows = {}
for d in sorted(glob.glob('/content/drive/MyDrive/SAA_results/phase_a_*')):
    level = os.path.basename(d).replace('phase_a_', '')
    p = f'{d}/csv/mvtec-indx-0.csv'
    if os.path.exists(p):
        rows[level] = pd.read_csv(p, index_col=0).mean(numeric_only=True)

# P3 = lan chay lite2 o Buoc 2
p3 = '/content/drive/MyDrive/SAA_results/run_lite2/csv/mvtec-indx-0.csv'
if os.path.exists(p3):
    rows['P3'] = pd.read_csv(p3, index_col=0).mean(numeric_only=True)

if rows:
    tab = pd.DataFrame(rows).T
    print(tab[[c for c in ('p_ap','p_f1','r_f1_fixed','t_dino','t_total') if c in tab]]
          .to_string(float_format='{:.2f}'.format))

    if {'P1', 'P3'} <= set(tab.index):
        gap = tab.loc['P3', 'p_f1'] - tab.loc['P1', 'p_f1']
        print(f'\nkhoang cach P1 -> P3: {gap:.2f} diem p_f1')
        for lvl in ('P2b', 'P2v'):
            if lvl in tab.index:
                closed = (tab.loc[lvl, 'p_f1'] - tab.loc['P1', 'p_f1']) / gap * 100
                mark = 'DAT' if closed >= 50 else 'chua dat'
                print(f'  {lvl}: thu hep {closed:.1f}%  {mark}  (spec muc 7 can >= 50%)')

    if 'P3' in tab.index:
        print('\n--- muc tieu 2: toc do ---')
        for lvl in tab.index:
            print(f'  {lvl:4s} t_dino {tab.loc[lvl, "t_dino"]:7.1f} ms  '
                  f't_total {tab.loc[lvl, "t_total"]:7.1f} ms  '
                  f'{tab.loc["P3", "t_total"] / tab.loc[lvl, "t_total"]:.2f}x so P3')
```

- [ ] **Step 3: Commit**

```bash
git add demo/Phase_A_LLM_Prompts.ipynb
git commit -m "feat: notebook for the Phase A prompt-source ladder"
```

---

## Ngan sach GPU

Cau hinh lite2: MVTec 0.92h, VisA 1.87h (do o Buoc 2).

| Muc | MVTec | VisA | Gio |
|---|---|---|---|
| P0 generic | 1 | — | 0.9 |
| P1 general | 1 | — | 0.9 |
| P2-blind | 1 | 1 | 2.8 |
| P2-vision | 1 | 1 | 2.8 |
| P3 manual | **da co** | **da co** | 0 |
| P3-clean (doi chung) | 1 | — | 0.9 |
| | | | **≈ 8.3h** |

P0 va P1 chi chay MVTec (spec muc 6.3): chung chi dong vai moc san, mat mat
khong dang ke.

Re hon nhieu so voi uoc tinh ban dau 18-22h, vi Phase B da giam mot nua chi phi
moi lan chay va P3 da co san.

## Tieu chi thanh cong

Spec muc 7, Phase A: **P2 vuot P1 ro ret tren da so class, va thu hep >= 50%
khoang cach tu P1 toi P3.**

Khong dat thi bao cao ket qua am kem phan tich class nao LLM truot va tai sao —
van la mot chuong hop le. Bo dem `prompt_box_counts` (commit `2554aac`) in ra
prompt nao cho 0 box tren toan class; do la du lieu cho phan phan tich do.

Muc tieu 2 khong co nguong chot truoc, vi no khong nam trong spec goc. Bao cao
`t_dino` va so prompt moi class, doi chieu voi P3.

## Rui ro

| Rui ro | Xu ly |
|---|---|
| LLM sinh cum ma DINO khong nam duoc -> 0 box | Bo dem `prompt_box_counts` da co. Prompt chet dua vao phan tich, khong giau |
| Ro ri du lieu qua duong doc anh | `train_image_paths` hard-code `train/good`, co test chan. Grep truoc khi commit |
| Confound dau phay lam sai lech so sanh P2-P3 | Lan chay doi chung P3-clean |
| JSON sai schema, lo ra giua lan chay dai | `load_prompt_file` validate luc nap; Task 5 Step 3 kiem truoc khi chay |
| Surface structured output cua `gpt-5.6-luna` khac du doan | Task 4 Step 1 probe mot lan goi truoc khi chay 54 |
| Khoa API lot vao git | `.env` da gitignore (dong 107), chua track. Task 4 Step 7 kiem lai |

## Notes

Dien trong luc thuc hien:

- So prompt trung binh moi class, mvtec-blind: ___
- So prompt trung binh moi class, mvtec-vision: ___
- So prompt cua P3 (thu cong) de doi chieu: ___
- Chi phi API thuc te: ___
- Prompt chet (0 box tren toan class): ___
