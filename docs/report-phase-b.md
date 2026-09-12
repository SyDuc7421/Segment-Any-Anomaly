# Bao cao Phase B — SAA-Lite

- **Ngay**: 2026-09-12
- **Trang thai**: Buoc 0, Buoc 1, Buoc 2 hoan thanh. Phase A: thang so sanh
  nguon prompt da chay (`results/phase_a_ladder/`), phan sinh prompt chua bat dau.
- **Phan cung**: Colab T4 (16 GB)
- **Spec**: `docs/superpowers/specs/2026-08-28-saa-lite-llm-prompt-design.md`

---

## 1. Tom tat

Nam ket qua, xep theo do vung chac.

**1. Tai lap duoc baseline tren MVTec.** max-F1-pixel do duoc 37.72 so voi 39.40
cua paper, lech 4.3%.

**2. VisA cao hon paper 24.6% — chua giai thich duoc.** Pipeline deterministic
nen day khong phai nhieu. Chua duoc dua vao luan van cho toi khi tim ra nguyen
nhan.

**3. Cai dat max-F1-region trong repo goc khong khop dinh nghia ma paper phat
bieu, va co the tra ve gia tri > 1.** Tai hien duoc bang mot vi du 5 dong.

**4. Thay SAM bang MobileSAM giu duoc gan nhu toan bo accuracy va nhanh gap
doi.** Tren full test set: MVTec giu 97.8% `p_ap` va 99.3% `p_f1`, nhanh 2.07x;
VisA thi lite2 con **tot hon** baseline (110.3% `p_ap`). Khong dat moc 3x cua
spec muc 7, nhung da vat 90% du dia ly thuyet cua truc SAM.

**5. Thay Grounding DINO bang YOLO-World hoac OWLv2: that bai hoan toan.** Ca
hai vua kem chinh xac hon (con 4-16%) vua cham hon. Khong co diem Pareto.

**6. Prompt thu cong cua tac gia thua chien luoc generic tren 6/15 class**, va
lam hai co he thong tren class texture. Chon dung nguon prompt cho tung class
vuot chuyen gia 2.33 diem `p_f1` ma khong can viet prompt moi nao. Chi tiet o
`results/phase_a_ladder/README.md`.

Ket luan chien luoc: sau khi thay SAM, **Grounding DINO chiem 69-72% thoi gian
va khong thay the duoc**. Giam so luot goi DINO — tuc Phase A — la don bay toc
do duy nhat con lai, va thang so sanh da xac nhan `t_dino` ty le thuan voi so
prompt (258-268 ms moi luot goi, on dinh tren 15 class).

---

## 2. Thiet lap

| Tham so | Gia tri |
|---|---|
| `eval_resolution` | 400 |
| `box_threshold` | 0.1 |
| `text_threshold` | 0.1 |
| `experiment_indx` | 0 (seed 111) |
| Tap anh | Full test set, khong dung `--max-samples` |
| `cal_pro` | Bat (nen co `p_pro`, `r_f1`, `r_f1_fixed`) |

Doi bat ky gia tri nao o tren dong nghia phai chay lai toan bo (spec muc 6.1).

**Rang buoc quan trong nhat**: `normalize()` trong `utils/eval_utils.py` la min-max
tren toan bo score cua ca class. Nen ket qua chay 34 anh va chay 117 anh
**khong so sanh duoc voi nhau**. Moi con so trong bao cao nay deu tu full test
set.

| Dataset | Class | Anh |
|---|---|---|
| MVTec-AD | 15 | 1725 |
| VisA (public, 1cls) | 12 | 2162 |

---

## 3. Buoc 0 — Tai lap baseline

Cau hinh goc: Grounding DINO Swin-T + SAM ViT-H + WideResNet-50, prompt thu cong
cua tac gia.

| | Do duoc | Paper | Lech |
|---|---|---|---|
| MVTec `p_f1` | 37.72 | 39.40 | −4.3% |
| MVTec `r_f1` | 42.45 | 49.67 | −14.5% |
| VisA `p_f1` | 33.74 | 27.07 | **+24.6%** |
| VisA `r_f1` | 15.95 | 14.46 | +10.3% |

### MVTec: tai lap thanh cong

Lech 4.3% tren `p_f1` la trong khoang chap nhan duoc cho mot lan tai lap doc lap.

### VisA: chua giai thich duoc

Day la diem yeu lon nhat cua bao cao nay, va can noi ro thay vi lam ngo.

Pipeline khong huan luyen gi, seed co dinh, nen chenh lech **khong the do nhieu**.
Mot dataset thap 4%, dataset kia cao 25%, **nguoc chieu nhau** — dau hieu cua
khac biet he thong chu khong phai sai so.

Gia thuyet chua loai tru:
- Paper danh gia tren split VisA khac (`2cls` / `4cls` thay vi `1cls`)
- Paper dung `visa_challenge` thay vi `visa_public`
- So anh test moi class khong khop voi bang trong paper

Cho toi khi tra loi duoc, **so VisA khong dua vao luan van nhu ket qua tai lap**.
"Ket qua cua em tot hon paper" khong phai cau tra loi duoc chap nhan neu khong
giai thich duoc tai sao.

---

## 4. Loi trong cai dat max-F1-region

Paper dinh nghia (muc 5.1):

> max-F1-region (F_r) ... we compute the F1-score for region-wise segmentation
> at the optimal threshold, considering a prediction positive if the overlapping
> value exceeds 0.6.

F1 theo dinh nghia nay **khong the vuot 1**: TP la mot con so duy nhat, dung
chung cho ca precision va recall, nen `TP <= min(n_pred, n_gt)`.

Nhung cai dat trong repo goc (`utils/metrics.py`, ham `calculate_max_f1_region`)
tra ve **122.57** cho class `wood`.

### Tai hien

```python
gt = np.zeros((1, 60, 60), dtype=np.uint8)
gt[0, 20:40, 20:40] = 1                    # mot vung GT lien khoi

scores = np.zeros((1, 60, 60))
scores[0, 20:40, 20:40] = 1.0
scores[0, 29:31, 20:40] = 0.0              # khe 1 pixel tach du doan lam doi

calculate_max_f1_region(gt, scores)        # -> 1.3333
```

Tren fixture nhieu Gauss, ham tra ve `recall = 2416.5`.

### Hai loi long nhau

**A — recall chia sai mau so.**
```python
for score_prop in score_props:      # lap theo vung DU DOAN
    pro.append(max(cur_pros))       # len(pro) == so vung du doan
hits = (pro >= pro_thresh).sum()
recall = hits / gt_region_number    # chia cho so vung GT
```
Tu so dem vung du doan, mau so la vung GT. Du doan vo thanh n manh cung trum
mot vung GT thi `hits` = n con mau so = 1.

**B — cua so tinh overlap bi nhiem.**
```python
cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
```
Cat **toan bo ban do nhi phan** theo bbox hop, khong phai rieng vung dang xet.
Moi vung khac roi vao cua so do deu gop vao ca giao lan hop. Day la ly do **ca
hai** manh deu dat IoU > 0.6.

Chinh tac gia de lai dau vet ngay cho do: `# cropped_mask = prop.filled_image  # corrected!`

### Cach xu ly: bao cao ca hai cot

| Cot | Y nghia |
|---|---|
| `r_f1` | Ban goc SAA+. Dung cot NAY de so voi 49.67, vi nhieu kha nang paper tinh bang chinh doan code do. Co the vuot 100. |
| `r_f1_fixed` | Cai dung dinh nghia paper phat bieu: TP tu phep ghep mot-mot, IoU tinh tren mat na tung vung. Luon <= 100. |

Ket qua:

| | `r_f1` | `r_f1_fixed` |
|---|---|---|
| MVTec | 42.45 | 21.92 |
| VisA | 15.95 | 11.78 |

Ban cu thoi phong khoang **2x** tren MVTec. Class `zipper` co `r_f1` = 10.91
trong khi `r_f1_fixed` = 0.00 — khong vung du doan nao dat IoU 0.6 voi vung GT
nao, toan bo 10.91 la ao.

Ban dung con **nhanh gap 25 lan** (3.9s so voi 98.6s tren 20 anh 400x400), vi bo
duoc vong cat bbox chong cheo.

---

## 5. Buoc 1 — Profiling

Mot class (`carpet`, full 117 anh), do latency tach theo giai doan.

### Truc SAM va saliency

| Cau hinh | SAM | Saliency | `p_ap` | `p_f1` | `t_dino` | `t_sam` | `t_saliency` | `t_total` | `peak_vram` |
|---|---|---|---|---|---|---|---|---|---|
| baseline | ViT-H | WideResNet-50 | 39.95 | 56.64 | 1837.5 | 2294.8 | 35.9 | 4255.4 | 6498.9 |
| lite1 | MobileSAM | WideResNet-50 | 36.48 | 55.53 | 1866.2 | 99.2 | 36.2 | 2095.7 | 1817.7 |
| lite2 | MobileSAM | MobileNetV3 | 37.16 | 55.36 | 1866.3 | 99.0 | 25.0 | 2077.9 | 1734.2 |

Don vi ms/anh va MB.

| | Thay doi |
|---|---|
| `t_sam` | 2294.8 → 99.0 — **23.2x** |
| `t_total` | 4255.4 → 2077.9 — **2.05x** |
| `peak_vram` | 6498.9 → 1734.2 — **3.75x** |
| `t_dino` | 1837.5 → 1866.3 — **0.98x**, khong doi |

### Ba ket luan

**1. Thay SAM an dut o tang cua no, nhung end-to-end bi DINO chan.**
Tran ly thuyet neu SAM va saliency ve 0: `4255.4 / 1866.3 = 2.28x`. Dat duoc
2.05x, tuc **da vat 90% du dia**. Khong con gi de lay them tu viec thay SAM.

**2. DINO tro thanh nut co chai** — xac nhan gia thuyet spec muc 4.4 bang so do:

| | dino | sam | saliency |
|---|---|---|---|
| baseline | 43% | 54% | 1% |
| lite2 | **90%** | 5% | 1% |

**3. Saliency khong phai cho dang toi uu.** No chiem 1% thoi gian; doi
WideResNet-50 sang MobileNetV3 tiet kiem 11 ms tren 2078 ms.

### Doi chieu tieu chi spec muc 7

| | Nguong | lite2 |
|---|---|---|
| Toc do | ≥ 3x | 2.05x ❌ |
| `p_ap` giu duoc | ≥ 95% | 93.0% ❌ |
| `p_f1` giu duoc | ≥ 95% | 97.7% ✅ |

**Khong dat.** Phuong an du phong da chot truoc khi chay (spec muc 7): xuat bang
Pareto va ket luan "SAA+ nen duoc toi dau truoc khi gay". Van la mot chuong hop le.

Dang chu y: VRAM giam 3.75x **vuot moc 3x**, chi la tren truc ma tieu chi khong
phu. Voi chuong ung dung day la con so quan trong hon ca toc do — SAA-Lite chay
lot GPU 2 GB, baseline thi khong.

### Ghi chu do luong

`carpet` trong lan full-run baseline co `t_total` = 4167.5, trong lan profiling
la 4255.4 — lech 2.1% giua hai lan chay cung cau hinh. Day la bien thien do
luong binh thuong tren GPU chia se; moi so sanh toc do trong bao cao nay deu lon
hon nguong do nhieu lan.

---

## 6. Truc detector — ket qua am

Mo rong ngoai pham vi spec ban dau, quyet dinh sau khi Buoc 1 cho thay DINO
chiem 90%. SAM va saliency khoa o lite2 de detector la bien duy nhat.

| Cau hinh | Detector | `p_ap` | `p_f1` | `t_dino` | `t_total` | `peak_vram` |
|---|---|---|---|---|---|---|
| lite2 | Grounding DINO | 37.16 | 55.36 | 1866.3 | 2077.9 | 1734.2 |
| lite2_owlv2 | OWLv2 | 4.59 | 9.22 | 3828.3 | 3981.8 | **923.8** |
| lite2_yolo | YOLO-World | 1.60 | 3.15 | 5463.6 | 5636.3 | 1950.4 |

**Ca hai bi chi phoi hoan toan**: vua kem chinh xac hon vua cham hon. Khong co
diem Pareto. YOLO-World con cham hon ca baseline ViT-H (0.76x) — mot model 13M
tham so thua model 636M ve toc do.

### Hai ket luan, ban chat khac nhau

**Accuracy sup la do model.** Con 4-16% khong phai danh doi, la hong. Grounding
DINO neo duoc cum mo ta defect (`"black hole"`, `"thread"`) vao vung anh.
YOLO-World va OWLv2 huan luyen de do **danh tu vat the**, khong do tinh tu mo ta
khuyet tat. Khong sua bang code duoc.

**Cham la do cach tich hop.** Pipeline goi detector mot luot moi prompt, 7 luot
moi anh, va ca hai deu ma hoa lai van ban moi luot. Gop cum lai thanh mot luot
se nhanh hon han — nhung khong dang lam, vi ke ca bang toc do DINO thi accuracy
van bi chi phoi.

Phai viet ro cho nay, neu khong hoi dong se hoi tai sao mot detector thoi gian
thuc lai cham hon SAM ViT-H.

### Diem sang duy nhat

OWLv2 co `peak_vram` = 923.8 MB, **7x nho hon baseline**. Khong cuu duoc accuracy
nhung la du kien that: cau hinh nhe nhat ve bo nho khong phai cau hinh nhe nhat
ve tham so.

---

## 6b. Buoc 2 — Full run lite2

Cau hinh lite2 (MobileSAM + MobileNetV3 + Grounding DINO) tren toan bo 27 class.
Chi tiet o `results/lite2_full/README.md`.

| | MVTec baseline | MVTec lite2 | Giu | VisA baseline | VisA lite2 | Giu |
|---|---|---|---|---|---|---|
| `p_ap` | 28.86 | 28.24 | 97.8% | 22.07 | 24.35 | **110.3%** |
| `p_f1` | 37.72 | 37.44 | 99.3% | 33.74 | 34.92 | **103.5%** |
| `r_f1_fixed` | 21.92 | 21.25 | 96.9% | 11.78 | 11.78 | 100.0% |
| `t_total` | 3994 | 1926 | **2.07x** | 4756 | 3110 | **1.53x** |
| Full run | 1.91h | 0.92h | | 2.86h | 1.87h | |

### Tieu chi spec muc 7

`p_ap` va `p_f1` **dat** tren ca hai dataset. Toc do **khong dat** (2.07x va
1.53x so voi moc 3x). Phuong an du phong da chot truoc khi chay: xuat bang
Pareto va ket luan "SAA+ nen duoc toi dau truoc khi gay".

### Ba dinh chinh cho ket luan Buoc 1

Buoc 1 do tren mot class (`carpet`). Full run cho thay ca ba ket luan deu lech.

**1. Tieu chi accuracy thuc ra DAT.** Buoc 1 bao 93.0% `p_ap`, truot moc 95%.
Tren 15 class la 97.8%. `carpet` tinh co la class lite2 mat nhieu hon trung
binh — uoc luong mot class da bi quan qua.

**2. "DINO chiem 90%" chi dung voi `carpet`.**

| | baseline | lite2 |
|---|---|---|
| `carpet` (Buoc 1) | 43% | 90% |
| MVTec toan bo | 34% | 72% |
| VisA toan bo | 40% | 69% |

`t_sam` bien thien manh theo class: 98 ms o `carpet`, 788 ms o `pill`. Decoder
SAM chay mot luot moi box, class nhieu vat the thi ton hon han.

**3. VRAM giam it hon Buoc 1 tuong.** `carpet` cho 3.75x; full run cho 1.86x
tren trung vi MVTec va 1.41x tren dinh. Dinh roi vao `pill`/`cable`/`pcb4`, noi
so box chi phoi bo nho chu khong phai trong so encoder.

### VisA: lite2 tot hon baseline

`p_ap` 110.3%, `p_f1` 103.5%. Cung tap anh, cung seed, chi khac SAM va saliency.

Gia thuyet: mask tho hon cua MobileSAM khop hon voi defect lon, mo ranh gioi.
Xu huong nay cung thay tren MVTec o muc class (`zipper` +9.4%, `capsule` +7.1%,
`cable` +5.9%), chi la tren VisA no du manh de lat ca trung binh. **Chua kiem
chung** — muon khang dinh phai doi chieu kich thuoc defect trung binh tung class
voi muc thay doi `p_f1`.

---

## 7. Bang tong hop

Tren `carpet`, 117 anh, T4.

| Cau hinh | `p_ap` | `p_f1` | `t_total` (ms) | Speedup | `peak_vram` (MB) |
|---|---|---|---|---|---|
| baseline | 39.95 | 56.64 | 4255.4 | 1.00x | 6498.9 |
| lite1 | 36.48 | 55.53 | 2095.7 | 2.03x | 1817.7 |
| **lite2** | **37.16** | **55.36** | **2077.9** | **2.05x** | **1734.2** |
| lite2_owlv2 | 4.59 | 9.22 | 3981.8 | 1.07x | 923.8 |
| lite2_yolo | 1.60 | 3.15 | 5636.3 | 0.76x | 1950.4 |

Full test set:

| | MVTec `p_f1` | MVTec `r_f1` | MVTec `r_f1_fixed` | VisA `p_f1` | VisA `r_f1` |
|---|---|---|---|---|---|
| baseline | 37.72 | 42.45 | 21.92 | 33.74 | 15.95 |
| **lite2** | **37.44** | **37.09** | **21.25** | **34.92** | **14.54** |
| paper | 39.40 | 49.67 | — | 27.07 | 14.46 |

---

## 8. Ket luan va viec con lai

### Da xac lap

- Baseline tai lap duoc tren MVTec
- MobileSAM la thay the tot: 2.07x nhanh hon tren MVTec, giu 97.8% `p_ap` va
  99.3% `p_f1`; tren VisA con vuot baseline
- Saliency backbone khong dang toi uu
- Grounding DINO khong thay the duoc bang YOLO-World hay OWLv2
- Cai dat max-F1-region goc co loi, da co ban dung ben canh
- `t_dino` ty le thuan voi so luot goi DINO: 258-268 ms moi luot, on dinh tren
  15 class va ba muc prompt khac nhau
- Prompt thu cong khong phai tran: thua generic tren 6/15 class, lam hai tren
  texture (-1.07 diem), giup tren object (+3.00 diem)

### Con mo

- **VisA cao hon paper 24.6%** — phai tra loi truoc khi dua so VisA vao luan van
- Vi sao lite2 **tot hon** baseline tren VisA — gia thuyet mask tho, chua kiem chung
- Vi sao `metal_nut` cho P1 == P3 chinh xac (36.13) — nghi prompt thu cong chet
- Phase A: phan sinh prompt bang LLM chua bat dau

### Don bay con lai

DINO chiem 72% tren MVTec (90% neu chi nhin `carpet`) va khong thay the duoc.
`carpet` goi DINO 7 luot moi anh
(`1 object + 3 general + 3 manual`), tuc **267 ms moi luot**.

**Da do, khong con la uoc tinh.** Thang so sanh chay tren full MVTec:

| Muc | Luot DINO/anh | `t_total` | Nhanh hon P3 |
|---|---|---|---|
| P0 | 2.00 | 735 ms | **2.62x** |
| P1 | 4.00 | 1505 ms | 1.28x |
| P3 | 5.20 | 1926 ms | 1.00x |

Giam so prompt keo `t_total` xuong dung nhu du doan. P0 dat 2.62x so voi P3 tren
15 class (va 3.14x tren rieng `carpet` o lat cat dau).

Nhung phat hien lon hon la **khong nguon prompt nao thong tri**: P0 thang 6
class, P1 thang 5, P3 thang 4. Chon dung nguon cho tung class cho 39.77 `p_f1`,
vuot P3 (37.44) 2.33 diem — ma khong viet mot prompt moi nao.

Phase A vi vay doi khung: cau hoi khong con la "LLM thu hep duoc bao nhieu
khoang cach toi oracle" (P3 khong phai oracle), ma la "LLM chon duoc chien luoc
prompt dung cho tung class khong". Du dia lon hon, va no la mot dong gop nham
thang vao tien de cua paper goc.

---

## Phu luc A — Loi phat hien duoc

Ba loi tiem an trong repo goc, ca ba deu chi lo ra khi di qua nhanh code ma
duong baseline khong bao gio cham toi.

| Loi | Vi tri | Trieu chung |
|---|---|---|
| `calculate_max_f1_region` recall > 1 | `utils/metrics.py` | `r_f1` = 122.57 cho class `wood` |
| Nhanh khong con box tra `list` thay vi `ndarray` | `SAA/model.py` | `TypeError: list indices must be integers or slices, not tuple` |
| `normalize()` chia cho 0 khi map la hang so | `utils/eval_utils.py` | `ValueError: Input contains NaN` sau 11 phut inference |

Hai loi cuoi chi lo ra khi thu detector khac. Do la gia tri phu cua ket qua am.

Ngoai ra: `eval_SAA.py` khai co `--vis` nhung truyen `is_vis=True` co dinh, nen
co do chua bao gio co tac dung.

## Phu luc B — Du lieu tho

| | |
|---|---|
| Baseline full-run | `results/baseline_full/` |
| Profiling Buoc 1 | `results/profiling_step1/` |
| Full-run lite2 (Buoc 2) | `results/lite2_full/` |
| Thang nguon prompt (Phase A so bo) | `results/phase_a_ladder/` |
| CSV tung cau hinh | Google Drive `SAA_results/prof_*/csv/` |
| Notebook baseline | `demo/Benchmark_SAA.ipynb` |
| Notebook profiling | `demo/Profiling_SAA_Lite.ipynb` |
| Notebook Buoc 2 | `demo/Benchmark_SAA_Lite2.ipynb` |

Moi lan chay deu co `run_meta.json` di kem ghi loai GPU, so anh, va cau hinh —
tru lan baseline dau tien, chay truoc khi co co che do.

**So latency chi so sanh duoc trong cung mot loai GPU.** Toan bo bao cao nay do
tren Colab T4.
