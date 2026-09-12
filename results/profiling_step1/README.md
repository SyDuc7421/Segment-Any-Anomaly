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

## Cau hinh thang cuoc: lite2

MobileSAM + MobileNetV3. Re hon tren moi truc: nhanh hon chut (2077.86 so voi
2095.70), VRAM thap hon (1734.22 so voi 1817.73), `p_ap` nhinh hon (93.0% so
voi 91.3%).

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
