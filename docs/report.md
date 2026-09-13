# SAA-Lite — Bao cao tong hop

- **Ngay**: 2026-09-13
- **Repo**: fork cua Segment-Any-Anomaly, nhanh `dev`
- **Phan cung**: Colab T4 (16 GB)
- **Spec**: `docs/superpowers/specs/2026-08-28-saa-lite-llm-prompt-design.md`

Do xem SAA+ nen duoc toi dau truoc khi gay, va prompt thu cong theo tung class
that su dong gop bao nhieu.

---

## 1. Tam ket qua

Xep theo do vung chac, khong theo thu tu lam.

**1. Tai lap duoc baseline tren MVTec.** max-F1-pixel 37.72 so voi 39.40 cua
paper, lech 4.3%.

**2. VisA cao hon paper 24.6% — bon nguyen nhan bi loai, mot con lai.** Khong
phai split, khong phai prompt bi sua, khong phai resize, khong rieng mot metric.
Chenh lech tap trung o 2/12 class.

**3. Cai dat max-F1-region cua repo goc khong khop dinh nghia paper phat bieu,
va co the tra ve gia tri > 1** (class `wood` ra 122.57). Tai hien duoc bang mot
vi du 5 dong.

**4. SAM khong phai nut co chai, Grounding DINO moi la.** Thay SAM ViT-H bang
MobileSAM lam tang do do 23x nhung end-to-end chi 2.07x. Ty trong DINO tang tu
34% len 72%.

**5. Grounding DINO khong thay the duoc.** YOLO-World va OWLv2 deu vua kem chinh
xac hon (con 4-16% `p_ap`) vua **cham hon**.

**6. Cau hinh lite2 giu gan nhu toan bo accuracy va nhanh gap doi.** MVTec giu
97.8% `p_ap` va 99.3% `p_f1`; tren VisA con vuot baseline.

**7. Prompt thu cong cua tac gia thua chien luoc generic tren 6/15 class**, va
lam hai co he thong tren texture (-1.07 diem) trong khi giup tren object (+3.00).

**8. Prompt cu the SAI nguy hiem hon prompt mo DUNG.** `pill` mat 52.84 diem khi
prompt la `crack`/`missing piece` thay vi mot chu `defect.`.

**9. LLM 3B that bai o ca hai bien the** (blind 31.16, vision 29.32, so voi 37.44
cua chuyen gia) **nhung ca hai deu nang tran khi la mot lua chon trong tap**.

**10. Cho model nhin anh binh thuong khien no mo ta tinh binh thuong lam prompt
defect.** `capsule` sinh ra `500` - lieu luong in tren vo nang - va sap 21.89 diem.

### Mot cau tom tat

Sau khi thay SAM, **Grounding DINO chiem 69-72% thoi gian va khong thay the
duoc**. So luot goi DINO ty le thuan voi so prompt (258-268 ms moi luot, on dinh
tren 15 class va nam nguon prompt), nen giam so prompt la don bay toc do duy
nhat con lai. Nhung khong nguon prompt nao thong tri: nam nguon, khong cai nao
thang qua 4/15 class, va **chon dung nguon cho tung class vuot chuyen gia 3.62
diem** ma khong viet mot prompt moi nao.

---

## 2. Thiet lap

| Tham so | Gia tri |
|---|---|
| `eval_resolution` | 400 |
| `box_threshold`, `text_threshold` | 0.1 |
| `experiment_indx` | 0 (seed 111) |
| Tap anh | Full test set, khong dung `--max-samples` |

Doi bat ky gia tri nao dong nghia phai chay lai toan bo (spec muc 6.1).

**Rang buoc quan trong nhat**: `normalize()` (`utils/eval_utils.py`) la min-max
tren toan bo score cua ca class, nen ket qua 34 anh va 117 anh **khong so sanh
duoc voi nhau**. Moi con so trong bao cao nay tu full test set.

| Dataset | Class | Anh |
|---|---|---|
| MVTec-AD | 15 | 1725 |
| VisA (public, 1cls) | 12 | 2162 |

**So latency chi so sanh duoc trong cung mot loai GPU.** Toan bo do tren T4.

---

## 3. Phase B — nen SAA+

### 3.1 Tai lap baseline

| | Do duoc | Paper | Lech |
|---|---|---|---|
| MVTec `p_f1` | 37.72 | 39.40 | -4.3% |
| MVTec `r_f1` | 42.45 | 49.67 | -14.5% |
| VisA `p_f1` | 33.74 | 27.07 | **+24.6%** |
| VisA `r_f1` | 15.95 | 14.46 | +10.3% |

MVTec coi nhu tai lap duoc. VisA thi khong, va day la diem yeu lon nhat can noi ro.

**Dieu tra VisA — bon gia thuyet bi loai:**

- **Khong phai split.** Loader mac dinh `VisA_pytorch/1cls`; so anh test khop ky
  vong cua 1cls, 11/12 class tuyet doi.
- **Khong phai prompt bi sua.** `visa_parameters.py` co dung mot commit, lan
  import goc.
- **Khong phai lech he thong.** Chenh lech o 2/12 class: `chewinggum` 86.12 va
  `capsules` 59.27. Bo hai class do, mean con 25.95 so voi 27.07 cua paper.
- **Khong phai resize, khong rieng mot metric.** `r_f1` lech cung huong; score va
  mask resize giong het nhau cho ca hai dataset.

**Con lai**: prompt trong repo co the khac prompt da tao ra Table 1. Paper day
chi tiet prompt sang supplementary material, khong co trong repo.

**Cach viet**: bao cao so do duoc, neu ro da loai tru gi, ghi rang chenh lech
tap trung o hai class. **Khong tuyen bo tai lap thanh cong tren VisA.**

### 3.2 Loi trong cai dat max-F1-region

Paper dinh nghia (muc 5.1): *"we compute the F1-score for region-wise
segmentation at the optimal threshold, considering a prediction positive if the
overlapping value exceeds 0.6"*. F1 theo dinh nghia do **khong the vuot 1**.

Cai dat trong repo tra ve **122.57** cho class `wood`.

```python
gt = np.zeros((1, 60, 60), dtype=np.uint8); gt[0, 20:40, 20:40] = 1
scores = np.zeros((1, 60, 60)); scores[0, 20:40, 20:40] = 1.0
scores[0, 29:31, 20:40] = 0.0              # khe 1 pixel tach du doan lam doi

calculate_max_f1_region(gt, scores)        # -> 1.3333
```

Hai loi long nhau: cua so tinh overlap cat **toan bo** ban do nhi phan theo bbox
hop nen moi vung khac deu gop vao; va tu so dem vung **du doan** khop duoc roi
chia cho so vung **GT**.

**Xu ly**: bao cao ca hai cot. `r_f1` (ban goc, de so voi bang da cong bo) va
`r_f1_fixed` (dung dinh nghia, TP tu phep ghep mot-mot). Ban cu thoi phong ~2x
tren MVTec; ban dung con nhanh gap 25 lan.

### 3.3 Buoc 1 — Profiling

Mot class (`carpet`, 117 anh), latency tach theo giai doan.

| Cau hinh | SAM | Saliency | `p_ap` | `p_f1` | `t_total` | `peak_vram` |
|---|---|---|---|---|---|---|
| baseline | ViT-H | WRN-50 | 39.95 | 56.64 | 4255.4 | 6498.9 |
| lite1 | MobileSAM | WRN-50 | 36.48 | 55.53 | 2095.7 | 1817.7 |
| **lite2** | MobileSAM | MobileNetV3 | 37.16 | 55.36 | **2077.9** | **1734.2** |

`t_sam` giam 23x nhung `t_total` chi 2.05x, vi `t_dino` khong doi. Tran ly thuyet
neu SAM va saliency ve 0 la 2.28x, nen **da vat 90% du dia**.

Ty trong DINO: baseline 43%, lite2 90% (tren rieng `carpet`; tren ca MVTec la 34%
va 72%). Gia thuyet spec muc 4.4 xac nhan bang so do.

Saliency chiem 1% thoi gian — **khong phai cho dang toi uu**.

### 3.4 Truc detector — ket qua am

Mo rong ngoai pham vi spec, quyet dinh sau khi thay DINO chiem 90%.

| Cau hinh | Detector | `p_ap` | `p_f1` | `t_total` | `peak_vram` |
|---|---|---|---|---|---|
| lite2 | Grounding DINO | 37.16 | 55.36 | 2077.9 | 1734.2 |
| lite2_owlv2 | OWLv2 | 4.59 | 9.22 | 3981.8 | **923.8** |
| lite2_yolo | YOLO-World | 1.60 | 3.15 | 5636.3 | 1950.4 |

**Ca hai bi chi phoi hoan toan.** YOLO-World con cham hon baseline ViT-H (0.76x).

Hai ket luan **khac ban chat**, khong duoc gop:

- **Accuracy sup la do model.** Grounding DINO neo duoc cum mo ta defect
  (`"black hole"`, `"thread"`); YOLO-World va OWLv2 huan luyen de do **danh tu
  vat the**. Khong sua bang code duoc.
- **Cham la do cach tich hop.** Pipeline goi detector mot luot moi prompt va ca
  hai deu ma hoa lai van ban moi luot. Gop lai se nhanh hon - nhung khong dang
  lam, vi ke ca bang toc do DINO thi accuracy van bi chi phoi.

OWLv2 co `peak_vram` 923.8 MB, **7x nho hon baseline** - diem sang duy nhat.

### 3.5 Buoc 2 — Full run lite2

| | MVTec baseline | MVTec lite2 | Giu | VisA baseline | VisA lite2 | Giu |
|---|---|---|---|---|---|---|
| `p_ap` | 28.86 | 28.24 | 97.8% | 22.07 | 24.35 | **110.3%** |
| `p_f1` | 37.72 | 37.44 | 99.3% | 33.74 | 34.92 | **103.5%** |
| `t_total` | 3994 | 1926 | **2.07x** | 4756 | 3110 | **1.53x** |
| Full run | 1.91h | 0.92h | | 2.86h | 1.87h | |

Accuracy **dat** nguong 95% tren ca hai dataset. Toc do **khong dat** moc 3x.

**Ba dinh chinh cho Buoc 1**, do no chi chay mot class:

| | `carpet` (Buoc 1) | Full MVTec |
|---|---|---|
| `p_ap` giu duoc | 93.0% — truot | **97.8%** — dat |
| DINO chiem | 90% | **72%** |
| VRAM giam | 3.75x | 1.86x trung vi |

Bai hoc phuong phap: **chon cau hinh bang mot class la rui ro**. Tin Buoc 1 thi
da ket luan lite2 truot tieu chi accuracy, trong khi thuc te no dat.

lite2 **tot hon** baseline tren VisA (110.3% `p_ap`) — chua giai thich duoc. Gia
thuyet: mask tho hon cua MobileSAM khop hon voi defect lon, mo ranh gioi.

---

## 4. Phase A — nguon prompt

Nam muc, chay tren full MVTec, cau hinh lite2.

| Muc | Nguon | Luot DINO/anh |
|---|---|---|
| P0 | `"defect."` | 2.00 |
| P1 | 3 `general_prompts` | 4.00 |
| P2b | Qwen2.5-VL-3B, chi biet ten class | 4.53 |
| P2v | Qwen2.5-VL-3B + 2 anh normal 512px | 3.40 |
| P3 | prompt thu cong cua tac gia | 5.20 |

### 4.1 So luot DINO ty le thuan voi so prompt

| Muc | Luot/anh | `t_dino` | ms/luot |
|---|---|---|---|
| P0 | 2.00 | 516.7 | 258.4 |
| P1 | 4.00 | 1050.7 | 262.7 |
| P2b | 4.53 | 1204.8 | 265.8 |
| P2v | 3.40 | 871.2 | 256.2 |
| P3 | 5.20 | 1394.1 | 268.1 |

On dinh qua nam nguon. `t_dino` chi phu thuoc SO LUOT goi, khong phu thuoc prompt
la gi. Don bay toc do la that.

### 4.2 P3 khong phai oracle

```
P1 thang P3 tren 6/15 class: grid, leather, tile, wood, bottle, toothbrush
texture   P3 - P1 = -1.07 diem     prompt thu cong LAM HAI
object    P3 - P1 = +3.00 diem     prompt thu cong giup
```

Co che: defect tren be mat van la "cho nao khac phan con lai", ma `"defect on
carpet"` dien ta dung the. Defect tren vat the co ten cu the ma prompt generic
khong goi ra duoc.

### 4.3 Ca hai bien the LLM that bai

| | `p_f1` | so voi P3 | Toc do |
|---|---|---|---|
| P2b blind | 31.16 | -6.28 | 1.26x |
| P2v vision | 29.32 | -8.12 | 1.70x |
| P3 | 37.44 | — | 1.00x |

Ca ba muc tieu chi deu khong dat, ca hai bien the.

**Prompt cu the SAI nguy hiem hon prompt mo DUNG.** `pill` mat **52.84 diem** khi
prompt sinh ra la `crack`/`scratch`/`missing piece` thay vi mot chu `"defect."`.
Defect `pill` trong MVTec la doi mau va in loi. `"defect."` khong ma hoa gia dinh
nao nen khong sai duoc.

**Cho nhin anh khien model mo ta tinh binh thuong.** Du doan truoc khi chay la
`pill` se tang, vi prompt vision la `red spots` va `letter F` - hai thu **co that**
tren anh. Thuc te giam (11.10 -> 8.08): chung co that tren **moi** vien thuoc, ke
ca vien binh thuong.

Cung co che do lam `capsule` sap 21.89 diem: prompt la `['500', 'capsule']`, trong
do `500` la lieu luong in tren vo nang.

> Cho model nhin anh **binh thuong** thi no mo ta **tinh binh thuong**. Anomaly
> detection can **cai khong nen co**.

Nhung P2v thang tuyet doi 2 class: `metal_nut` 40.77 (cao nhat trong ca nam nguon,
prompt la `cracked`/`deformed`) va `toothbrush` 11.79.

### 4.4 Khong nguon nao thong tri — ket qua trung tam

```
Nguon thang o tung class:  P1 4   P0 3   P2b 3   P3 3   P2v 2
```

| Chon giua | `p_f1` | Vuot P3 |
|---|---|---|
| P3 mot minh | 37.44 | — |
| oracle-3 (P0/P1/P3) | 39.77 | +2.33 |
| oracle-4 (+P2b) | 40.80 | +3.36 |
| **oracle-5 (+P2v)** | **41.06** | **+3.62** |

Ca hai bien the LLM **that bai khi dung dai tra** nhung **deu nang tran khi la
mot lua chon trong tap**. Bien thien theo class rat lon: `pill` co P0 hon P3 19
diem, `cable` co P3 hon P0 18 diem.

---

## 5. Bang tong hop

Full MVTec, cau hinh lite2, `p_f1`:

| Nguon | `p_f1` | `t_total` | Nhanh hon P3 |
|---|---|---|---|
| P0 generic | 35.54 | 735 | **2.62x** |
| P1 general | 35.80 | 1505 | 1.28x |
| P2b blind | 31.16 | 1527 | 1.26x |
| P2v vision | 29.32 | 1131 | 1.70x |
| P3 thu cong | **37.44** | 1926 | 1.00x |
| oracle-5 | **41.06** | — | — |

Backbone, tren `carpet`:

| Cau hinh | `p_f1` | Speedup | `peak_vram` |
|---|---|---|---|
| baseline | 56.64 | 1.00x | 6498.9 |
| **lite2** | 55.36 | **2.05x** | **1734.2** |
| lite2_owlv2 | 9.22 | 1.07x | 923.8 |
| lite2_yolo | 3.15 | 0.76x | 1950.4 |

---

## 6. Dong gop

Bon phat hien nham thang vao tien de cua paper goc, khong phu thuoc vao viec LLM
lam duoc gi:

1. **Nut co chai la Grounding DINO, khong phai SAM** - nguoc voi truc giac khi
   nhin kich thuoc model (636M so voi 172M tham so).
2. **Prompt thu cong theo tung class - thu paper trinh bay nhu doi hoi kien thuc
   chuyen gia - thua chien luoc generic tren 6/15 class** va lam hai co he thong
   tren texture.
3. **Cai dat max-F1-region cua repo khong khop dinh nghia paper phat bieu** va co
   the tra ve gia tri > 1, co vi du tai hien toi thieu.
4. **Chon dung nguon prompt cho tung class vuot chuyen gia 3.62 diem** ma khong
   viet mot prompt moi nao.

Hai phat hien ve ban chat prompt trong ZSAS:

5. **Prompt cu the sai nguy hiem hon prompt mo dung** (`pill`, -52.84 diem).
6. **Dieu kien hoa bang anh binh thuong khien model mo ta tinh binh thuong**
   (`capsule` -> `500`).

Mot bai hoc phuong phap:

7. **Chon cau hinh bang mot class la rui ro** - ca ba ket luan rut ra tu `carpet`
   deu lech khi kiem lai tren 15 class.

---

## 7. Con mo, va chua lam

**Ba cau hoi chua tra loi duoc:**

- VisA cao hon paper 24.6%. Bon nguyen nhan da bi loai; con lai la kha nang prompt
  trong repo khac prompt tao ra Table 1. Can supplementary material cua tac gia.
- lite2 **tot hon** baseline tren VisA (110.3% `p_ap`). Gia thuyet mask tho, chua
  kiem chung.
- `metal_nut` cho P1 == P3 chinh xac 36.13. Nghi prompt thu cong chet (0 box song
  sot). Bo dem da co, chua chay lai de lay so.

**Chua chay:**

- Kiem bit-exact bang `git stash` (muc 2.2 `docs/superpowers/COLAB_HANDOFF.md`).
  Viec factory backbone khong doi hanh vi hien moi duoc xac minh bang doc code.
- Phase A tren VisA.
- Web demo (spec muc 9).

---

## Phu luc A — Loi phat hien trong repo goc

Bon loi tiem an. Ba cai cuoi chi lo ra khi di qua nhanh code ma duong baseline
khong bao gio cham toi.

| Loi | Vi tri | Trieu chung |
|---|---|---|
| `calculate_max_f1_region` recall > 1 | `utils/metrics.py` | `r_f1` = 122.57 cho `wood` |
| Nhanh khong con box tra `list` thay vi `ndarray` | `SAA/model.py` | `TypeError: list indices must be integers` |
| `normalize()` chia cho 0 khi map hang so | `utils/eval_utils.py` | `ValueError: Input contains NaN` sau 11 phut |
| `--vis` khai nhung `is_vis=True` co dinh | `eval_SAA.py` | co chua bao gio co tac dung |

## Phu luc B — Du lieu va tai lap

| | |
|---|---|
| Baseline full-run | `results/baseline_full/` |
| Profiling Buoc 1 + truc detector | `results/profiling_step1/` |
| Full-run lite2 (Buoc 2) | `results/lite2_full/` |
| Thang nguon prompt (Phase A) | `results/phase_a_ladder/` |
| Prompt do LLM sinh | `SAA/prompts/generated/` |
| Ket qua subset 34 anh (KHONG dung duoc) | `results/subset34/` |

Notebook: `demo/Benchmark_SAA.ipynb` (baseline), `Profiling_SAA_Lite.ipynb`
(Buoc 1), `Benchmark_SAA_Lite2.ipynb` (Buoc 2), `Phase_A_Prompt_Ladder.ipynb`
(thang prompt), `Generate_Prompts_VLM.ipynb` va `Generate_Prompts_VLM_Vision.ipynb`
(sinh prompt).

Prompt do LLM sinh **tai lap duoc 100%**: model trong so mo chay cuc bo, greedy
decoding, khong goi API ngoai. `*-meta.json` ghi model, quantization, so anh,
kich thuoc anh, va nguyen van prompt da dung.

Moi lan chay benchmark deu co `run_meta.json` ghi loai GPU, so anh va cau hinh -
tru lan baseline dau tien, chay truoc khi co co che do.

## Phu luc C — Nhiem du lieu huan luyen

SAA+ la repo cong khai, paper dang IEEE. VLM co the da thay prompt cua tac gia
trong du lieu huan luyen. Doi chieu tung chu:

```
blind    0/53 prompt trung khit voi prompt thu cong
vision   0/36
```

Khong co dau hieu nho truc tiep.
