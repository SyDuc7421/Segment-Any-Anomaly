# Thay Grounding DINO — mo rong pham vi Phase B

**Goal:** Do xem thay Grounding DINO bang detector open-vocabulary khac lam
metric tut bao nhieu va nhanh len bao nhieu, khi DINO da chiem 90% thoi gian
cua cau hinh lite2.

**Spec:** `docs/superpowers/specs/2026-08-28-saa-lite-llm-prompt-design.md`

**Vi tri:** mo rong **ngoai pham vi spec ban dau**. Spec muc 3 (phi muc tieu 3)
chot Phase B chi thay SAM va saliency; muc 10 xep YOLO-World va OWLv2 vao
Related Work. Nguoi dung quyet dinh mo rong sau khi Buoc 1 cho thay DINO chiem
90% thoi gian, tuc la don bay con lai duy nhat.

Chay **truoc Buoc 2**, de neu co cau hinh detector nao dang gia thi Buoc 2 do
luon thay vi phai chay lai.

## Vi sao day khong phai drop-in nhu MobileSAM

MobileSAM giu nguyen interface `SamPredictor` nen thay la xong. DINO thi khong:
`bbox_suppression` (`SAA/model.py:230`) an sau vao ba thu rieng cua DINO.

```python
logits = outputs["pred_logits"].sigmoid()[0]     # (nq, 256) — diem theo TUNG TOKEN
tokenlizer = self.anomaly_region_generator.tokenizer
pred_phrase = get_phrases_from_posmap(logit > text_score_thr, tokenized, tokenlizer)
if pred_phrase.count(filtered_phrase) > 0:       # strategy5: loc nen
    continue
```

`logits` khong phai mot diem tin cay moi box — no la ma tran `(so query x 256
token BERT)`. Pipeline dung no de hoi nguoc: *box nay khop voi chu nao trong
cau prompt?* Roi strategy5 vut bo box nao khop voi `filter_prompt` (vi du
`'carpet'`) thay vi khop voi defect.

YOLO-World va OWLv2 chi tra ve **mot diem moi (box, query)**. Khong co ma tran
token, khong co tokenizer tuong ung.

Anh xa tu nhien: thay vi mot cau prompt gop, truyen tung cum lam **query
rieng**. Khi do "box nay khop cum nao" = argmax theo query — tuong duong
strategy5 ma khong can ma tran token.

Luu y: repo **da dung ban DINO nho nhat** (`GroundingDINO_SwinT_OGC`). Khong co
bac nhe hon trong cung ho, nen day la doi kien truc chu khong phai ha co.

## Rang buoc quan trong nhat

**Duong DINO khong duoc doi mot dong nao.** Baseline phai giu bit-exact (spec
muc 3). Nen thiet ke la **nhanh song song**, khong phai refactor:

- `text_guided_region_proposal` va `bbox_suppression` giu nguyen, chi chay khi
  `detector == 'grounding_dino'`.
- Detector kieu diem-theo-query di qua ham moi `query_detector_proposal`, tra
  ve dung bo ba `(boxes_filtered, logits_filtered, pred_phrases)` ma phan sau
  cua pipeline dang cho.
- `ensemble_text_guided_mask_proposal` chi them mot nhanh `if` o dau vong lap.

Hop dong dau ra, giong het duong DINO dang tra:

| | Kieu | Ghi chu |
|---|---|---|
| `boxes_filtered` | tensor `(n, 4)` tren device | **cxcywh, chuan hoa [0,1]** — phia sau tinh `boxes[:, 2] * boxes[:, 3]` lam dien tich roi moi denormalize |
| `logits_filtered` | list float | diem tin cay moi box |
| `pred_phrases` | list str | cum tu box khop, dung cho phan tich prompt chet |

Tra sai he toa do la loi im lang: dien tich sai keo theo `defect_max_area` sai,
va anomaly map van ra so nhin co ve hop ly.

## File thay doi

| File | Thay doi |
|---|---|
| `SAA/detectors.py` | **Moi.** `DETECTORS`, `build_detector(name, device)`, adapter cho yolo_world va owlv2 |
| `SAA/model.py` | `__init__` nhan `detector='grounding_dino'`; them `query_detector_proposal`; mot nhanh `if` trong `ensemble_text_guided_mask_proposal` |
| `eval_SAA.py` | Them `--detector` |
| `run_MVTec.py`, `run_VisA_public.py` | Doc `DETECTOR` tu bien moi truong, dua vao run_identity |
| `tests/test_detectors.py` | **Moi.** Bang ten, duong loi, chuyen doi toa do |
| `demo/Profiling_SAA_Lite.ipynb` | Them cai dat va luoi cau hinh detector |

Khong dung toi `utils/metrics.py`, `utils/timing.py`, `SAA/backbones.py`.

## Task 1: `SAA/detectors.py`

**Files:** Create `SAA/detectors.py`, `tests/test_detectors.py`

Cung rang buoc nhu `SAA/backbones.py`: **khong import nang o cap module**, moi
import nam trong than ham. Nho vay test nap file theo duong dan chay duoc tren
may khong co torch, va thieu ultralytics chi lam hong nhanh yolo_world chu
khong hong baseline.

```
DETECTORS = {
    'grounding_dino': ...,   # mac dinh, di duong cu
    'yolo_world': ...,
    'owlv2': ...,
}
QUERY_SCORED = {'yolo_world', 'owlv2'}   # loai tra diem moi (box, query)
```

Moi adapter co mot method:

```python
def detect(self, image_bgr, phrase, score_thr):
    """Tra ve (boxes_cxcywh_norm, scores, phrases) cho MOT cum prompt."""
```

Tach cum: prompt trong repo dang dang `'blue defect. black defect. scratch.'`.
DINO nuot ca cau; detector kieu query can tach theo `.` thanh nhieu class rieng.
Ham `split_phrase(phrase)` lam viec do, co test.

**Buoc TDD:**
1. Test bang ten `DETECTORS` va `QUERY_SCORED`, thong bao loi khi ten sai.
2. Test `split_phrase`: `'blue defect. black defect.'` -> `['blue defect', 'black defect']`; xu ly khoang trang thua va dau cham cuoi.
3. Test `xyxy_to_cxcywh_norm`: box pixel `(10, 20, 30, 60)` tren anh `100x200` -> `(0.2, 0.2, 0.2, 0.2)`. **Day la test quan trong nhat cua task** — sai he toa do la loi im lang.
4. Cai dat.

## Task 2: Noi vao `SAA/model.py`

**Files:** Modify `SAA/model.py`

`__init__` nhan `detector='grounding_dino'`. Khi la `grounding_dino` thi dung
`load_dino` nhu cu, khong doi gi. Nguoc lai goi `build_detector`.

Them `query_detector_proposal(image, phrase, filtered_phrase, score_thr,
object_max_area, object_min_area)`:

1. `terms = split_phrase(phrase)`
2. `boxes, scores, phrases = self.detector.detect(image, terms, score_thr)`
3. Loc dien tich: `boxes[:, 2] * boxes[:, 3]` trong khoang `(min, max)` — **cung
   cong thuc voi `bbox_suppression`**, khong viet lai theo cach khac.
4. Loc nen: bo box nao co `filtered_phrase` trong `phrases` — tuong duong
   strategy5.
5. Khong con box nao thi tra `(None, None, None)`, giong `bbox_suppression`.

Trong `ensemble_text_guided_mask_proposal`, them nhanh o dau vong lap. Than
vong lap hien tai giu nguyen trong nhanh `grounding_dino`.

Moc do gio `self.timer.stage('dino')` phai bao ca nhanh moi, neu khong cot
`t_dino` cua cau hinh detector se bang 0 va bang Pareto vo nghia.

## Task 3: CLI va runner

**Files:** Modify `eval_SAA.py`, `run_MVTec.py`, `run_VisA_public.py`

- `--detector` voi `choices` khop `DETECTORS`, mac dinh `grounding_dino`.
- Runner doc `DETECTOR` tu bien moi truong.
- **`detector` phai vao `run_identity`**: doi detector la doi con so, ket qua cu
  khong duoc coi la resume duoc.

## Task 4: Notebook

**Files:** Modify `demo/Profiling_SAA_Lite.ipynb`

Cell cai dat rieng cho `ultralytics` (YOLO-World). OWLv2 di qua `transformers`
da co san nen khong them dependency — day la ly do chon no, sau vu EfficientViT
treo 40 phut vi pip backtracking.

Luoi profiling mo rong. Detector chi so sanh duoc khi SAM va saliency giu co
dinh, nen khoa ca hai o cau hinh lite2 da thang Buoc 1:

| Ten | Detector | SAM | Saliency |
|---|---|---|---|
| lite2 | grounding_dino | MobileSAM | MobileNetV3 |
| lite2_yolo | yolo_world | MobileSAM | MobileNetV3 |
| lite2_owlv2 | owlv2 | MobileSAM | MobileNetV3 |

Moi cau hinh mot thu muc Drive rieng, skip cau hinh da co so — giong luoi hien
tai.

## Ky vong, ghi truoc khi chay

**Metric se tut, va do la ket qua can bao cao chu khong phai that bai.**
Grounding DINO manh o cho hieu cum tu mo ta defect truu tuong (`"black hole"`,
`"thread"`). YOLO-World toi uu cho danh tu vat the thuong gap. Cau hoi la tut
bao nhieu, doi lay bao nhieu toc do — dung tinh than Pareto.

Rui ro cu the: prompt sinh ra cum ma detector khong nam duoc -> khong box nao
song sot -> anomaly map rong. Dem so box con lai sau khi loc, prompt nao cho 0
box tren toan class la prompt chet. Dua vao phan tich, khong giau di.

**OWLv2 khong nhe hon DINO Swin-T.** Chay no de so accuracy, khong phai de
nhanh hon. Neu no cham hon that thi day cung la mot diem tren Pareto.

## Notes

Dien trong luc thuc hien:

- API `post_process_grounded_object_detection` cua OWLv2 tren transformers 5.x: ___
- YOLO-World: `set_classes` co ton thoi gian moi lan goi khong: ___
- So box song sot sau loc, so voi Grounding DINO: ___
