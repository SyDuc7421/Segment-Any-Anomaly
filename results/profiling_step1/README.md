# Buoc 1 — Profiling 4 cau hinh SAA-Lite

- **Ngay**: 2026-09-12
- **Giao thuc**: spec muc 6.2, Buoc 1
- **Class**: `carpet`, full 117 anh (khong dung `--max-samples`)
- **GPU**: Colab T4
- **Co**: `--cal-pro False`, `--vis False` (profiling chi can latency)
- **Tham so khoa cung**: `eval_resolution=400`, `box_threshold=0.1`, `text_threshold=0.1`, `experiment_indx=0`

Lite-3 (EfficientViT-SAM-L0) **khong chay duoc**: `pip install` roi vao vong
backtracking cua resolver, treo hon 40 phut khong hoi tu. Bo theo dung phuong
an du phong da ghi trong plan. Hai cau hinh Lite con lai du de ve Pareto.

## Ket qua

| Cau hinh | SAM | Saliency | `p_ap` | `p_f1` | `t_dino` | `t_sam` | `t_saliency` | `t_total` | `peak_vram` |
|---|---|---|---|---|---|---|---|---|---|
| baseline | ViT-H | WideResNet-50 | 39.95 | 56.64 | 1837.52 | 2294.81 | 35.88 | 4255.43 | 6498.87 |
| lite1 | MobileSAM | WideResNet-50 | 36.48 | 55.53 | 1866.19 | 99.22 | 36.16 | 2095.70 | 1817.73 |
| lite2 | MobileSAM | MobileNetV3 | 37.16 | 55.36 | 1866.31 | 99.04 | 25.01 | 2077.86 | 1734.22 |

Don vi: ms/anh, MB.

## Doi chieu tieu chi spec muc 7

| | Nguong | lite1 | lite2 |
|---|---|---|---|
| Toc do | >= 3x | 2.03x | 2.05x |
| `p_ap` giu duoc | >= 95% | 91.3% | 93.0% |
| `p_f1` giu duoc | >= 95% | 98.0% | 97.7% |

**Khong dat.** Theo phuong an du phong da chot truoc khi chay (spec muc 7):
van xuat bang Pareto va ket luan "SAA+ nen duoc toi dau truoc khi gay".

## Bon ket luan

### 1. Thay SAM an dut o tang cua no, nhung end-to-end bi DINO chan

`t_sam` giam 2294.81 -> 99.04 ms, tuc **23x**. Nhung `t_total` chi giam 2.05x,
vi `t_dino` khong doi (1837.52 -> 1866.31).

Tran ly thuyet neu SAM va saliency ve 0: `4255.43 / 1866.31 = 2.28x`. Dat duoc
2.05x, tuc **da vat 90% du dia**. Khong con gi de lay them tu viec thay SAM.

### 2. Gia thuyet spec muc 4.4 duoc xac nhan bang so do

Ty trong DINO trong tong thoi gian:

| | dino | sam | saliency |
|---|---|---|---|
| baseline | 43% | 54% | 1% |
| lite1 | 89% | 5% | 2% |
| lite2 | 90% | 5% | 1% |

Sau khi thay SAM, **DINO tro thanh nut co chai**, dung nhu du doan. Day khong
con la gia thuyet.

### 3. VRAM giam 3.7x — vuot moc 3x, chi la tren truc khac

`peak_vram` 6498.87 -> 1734.22 MB. SAA-Lite chay lot GPU 2 GB, baseline thi
khong. Voi chuong ung dung (web demo) day la con so quan trong hon ca toc do.

### 4. Saliency khong phai cho dang toi uu

Saliency chiem 1% thoi gian. Doi WideResNet-50 sang MobileNetV3 tiet kiem 11 ms
tren tong 2078 ms. Ban than dieu do la mot phat hien: nhanh nhat khong nam o do.

## Truc detector — ket qua am, dut khoat

Mo rong ngoai pham vi spec ban dau, quyet dinh sau khi thay DINO chiem 90%
thoi gian. Xem `docs/superpowers/plans/2026-09-12-detector-swap.md`.

SAM va saliency khoa o lite2 de detector la bien duy nhat.

| Cau hinh | Detector | `p_ap` | `p_f1` | `t_dino` | `t_total` | `peak_vram` |
|---|---|---|---|---|---|---|
| lite2 | Grounding DINO | 37.16 | 55.36 | 1866.31 | 2077.86 | 1734.22 |
| lite2_owlv2 | OWLv2 | 4.59 | 9.22 | 3828.34 | 3981.79 | 923.78 |
| lite2_yolo | YOLO-World | 1.60 | 3.15 | 5463.62 | 5636.28 | 1950.43 |

**Ca hai deu bi chi phoi hoan toan**: vua kem chinh xac hon, vua cham hon. Khong
co diem Pareto nao. YOLO-World con cham hon ca baseline ViT-H (0.76x) - mot
model 13M tham so thua model 636M ve toc do.

### Hai ket luan, ban chat khac nhau - khong duoc gop

**Accuracy sup la do model.** 4-16% cua baseline khong phai danh doi, la hong.
Grounding DINO neo duoc cum mo ta defect (`"black hole"`, `"thread"`) vao vung
anh. YOLO-World va OWLv2 huan luyen de do **danh tu vat the**, khong do tinh tu
mo ta khuyet tat. Khong sua bang code duoc.

**Cham la do cach tich hop, khong phai do model.** Pipeline goi detector mot
luot moi prompt, 7 luot moi anh, va ca hai deu **ma hoa lai van ban moi luot**:
`set_classes()` cua YOLO-World chay CLIP text encoder, OWLv2 chay lai text
tower. DINO cung encode text moi luot nhung re hon nhieu.

Gop tat ca cum thanh mot luot goi moi anh se nhanh hon han. **Nhung khong dang
lam**: ke ca bang toc do DINO thi accuracy van 4-16%, van bi chi phoi. Sua toc
do khong cuu duoc ket qua.

Phai viet ro cho nay trong luan van, neu khong hoi dong se hoi tai sao mot
detector thoi gian thuc lai cham hon SAM ViT-H.

### Diem sang duy nhat: VRAM

`peak_vram` cua lite2_owlv2 la 923.78 MB, **7x nho hon baseline**. Khong cuu
duoc accuracy nhung la du kien that cho bang Pareto: cau hinh nhe nhat ve bo
nho khong phai cau hinh nhe nhat ve tham so.

### Hai loi tiem an lo ra nho thi nghiem nay

Ca hai co tu truoc, chua bao gio lo vi Grounding DINO luon do duoc thu gi do:

1. `SAA/model.py` nhanh khong con box nao tra `list` trong khi nhanh thanh cong
   tra `ndarray`. `visual_saliency_calculation` index bang `masks[i, :, :]` nen
   gay `TypeError`. Sua o commit `2554aac`.
2. `utils/eval_utils.py` `normalize()` chia cho 0 khi map la hang so, cho ra NaN
   toan bo va lam `metric_cal` gay sau 11 phut inference. Sua o commit `69d6e58`.

Day la gia tri phu cua ket qua am: no di qua nhung nhanh code ma duong baseline
khong bao gio cham toi.

## Cau hinh thang cuoc: lite2

MobileSAM + MobileNetV3, **giu Grounding DINO**. Re hon lite1 tren moi truc:
nhanh hon chut (2077.86 so voi 2095.70), VRAM thap hon (1734.22 so voi
1817.73), `p_ap` nhinh hon (93.0% so voi 91.3%). Va bo xa hai cau hinh doi
detector tren ca hai truc.

**Canh bao khi doc con so nay**: day la MOT class. Chenh lech `p_ap` 36.48 so
voi 37.16 tren rieng `carpet` khong du de xep hang chac chan. Chon lite2 vi no
khong thua o dau ca, khong phai vi no thang thuyet phuc. Buoc 2 chay full moi
xac nhan duoc.

## Cong cache DINO: bo han

Uoc full MVTec (1725 anh):

| | |
|---|---|
| baseline | 2.0h |
| lite1 | 1.0h |
| lite2 | 1.0h |

Duoi nguong 2h cua spec muc 4.5, nen **khong them cache DINO**. Bot duoc mot
mang phuc tap.

## He qua cho Phase A

DINO chiem 90% thoi gian. `carpet` co `1 object + 3 general + 3 manual = 7`
luot DINO moi anh, tuc **267 ms moi luot**.

Nghia la Phase A khong con chi la chuyen accuracy — no la **don bay toc do duy
nhat con lai**. Neu LLM sinh 3 prompt trung thay vi 6:

```
4 luot DINO x 267 = 1067 ms
t_total ~ 1067 + 99 + 25 = 1191 ms
speedup so baseline = 3.57x     <- vuot moc >= 3x
```

Day la uoc tinh, chua do. Nhung no cho thay hai phase ke chung mot cau chuyen
thay vi la hai chuong roi rac, dung nhu spec muc 4.4 mong doi.

## Du lieu tho

CSV tung cau hinh nam tren Google Drive: `SAA_results/prof_{baseline,lite1,lite2}/csv/mvtec-indx-0.csv`.
