# Buoc 2 — Full run cau hinh lite2

- **Ngay**: 2026-09-12
- **Cau hinh**: `detector=grounding_dino`, `sam_variant=mobile_sam`, `saliency_backbone=mobilenetv3`, prompt thu cong
- **Tap anh**: full test set (MVTec 15 class / 1725 anh, VisA public 1cls 12 class / 2162 anh)
- **GPU**: Tesla T4 (moi class, xac nhan tu `run_meta.json`)
- **`cal_pro`**: bat
- **Tham so khoa cung**: `eval_resolution=400`, `box_threshold=0.1`, `text_threshold=0.1`, `experiment_indx=0`

Lan chay nay co `run_meta.json` di kem — 27 entry, mot cau hinh duy nhat,
`max_samples: None`. Khac lan baseline dau tien, provenance day du.

## Ket qua

### MVTec-AD

| | baseline | lite2 | Giu duoc |
|---|---|---|---|
| `p_ap` | 28.86 | 28.24 | **97.8%** |
| `p_f1` | 37.72 | 37.44 | **99.3%** |
| `r_f1_fixed` | 21.92 | 21.25 | 96.9% |
| `r_f1` | 42.45 | 37.09 | 87.4% |
| `p_pro` | 42.75 | 42.10 | 98.5% |
| `t_total` | 3994 ms | 1926 ms | **2.07x nhanh** |
| `peak_vram` (trung vi) | 6499 MB | 3495 MB | 1.86x nho |
| Full run | 1.91h | 0.92h | |

### VisA

| | baseline | lite2 | Giu duoc |
|---|---|---|---|
| `p_ap` | 22.07 | 24.35 | **110.3%** |
| `p_f1` | 33.74 | 34.92 | **103.5%** |
| `r_f1_fixed` | 11.78 | 11.78 | 100.0% |
| `r_f1` | 15.95 | 14.54 | 91.2% |
| `p_pro` | 36.96 | 37.28 | 100.9% |
| `t_total` | 4756 ms | 3110 ms | **1.53x nhanh** |
| `peak_vram` (trung vi) | 6499 MB | 2262 MB | 2.87x nho |
| Full run | 2.86h | 1.87h | |

## Doi chieu tieu chi spec muc 7

| | Nguong | MVTec | VisA |
|---|---|---|---|
| `p_ap` giu duoc | >= 95% | 97.8% OK | 110.3% OK |
| `p_f1` giu duoc | >= 95% | 99.3% OK | 103.5% OK |
| `r_f1` khong tut qua 5 diem | 5 diem | tut 5.36 — sat nguong | tut 1.41 OK |
| Toc do | >= 3x | 2.07x KHONG | 1.53x KHONG |

Accuracy dat. Toc do khong. Phuong an du phong da chot truoc khi chay (spec muc
7): xuat bang Pareto va ket luan "SAA+ nen duoc toi dau truoc khi gay".

Luu y ve dong `r_f1`: tut 5.36 diem tren MVTec la sat nguong 5 diem, nhung cot
do la ban cai dat co loi (xem `docs/report-phase-b.md` muc 4). `r_f1_fixed` chi
tut 0.67 diem. Neu xet theo cai dat dung dinh nghia thi tieu chi nay dat thoai mai.

## Ba dinh chinh cho ket luan Buoc 1

Buoc 1 do tren mot class (`carpet`). Full run cho thay ba ket luan do bi lech.

### 1. Tieu chi accuracy THUC RA DAT

Buoc 1 bao `p_ap` giu duoc 93.0%, truot moc 95%. Tren ca 15 class la **97.8%**.
`carpet` tinh co la class lite2 mat nhieu hon muc trung binh. Uoc luong mot class
da bi quan qua.

### 2. "DINO chiem 90%" chi dung voi `carpet`

| | baseline | lite2 |
|---|---|---|
| `carpet` (Buoc 1) | 43% | 90% |
| MVTec (toan bo) | 34% | **72%** |
| VisA (toan bo) | 40% | **69%** |

Vi `t_sam` bien thien rat manh theo class: 98 ms o `carpet`, 788 ms o `pill`.
Decoder cua SAM chay mot luot moi box, class nhieu vat the thi ton hon han. DINO
van ap dao nhung khong toi muc 90%.

### 3. VRAM giam it hon Buoc 1 tuong

Buoc 1 tren `carpet`: 3.75x. Full run: **1.86x** tren trung vi MVTec, **1.41x**
tren dinh. Dinh cua ca hai deu roi vao `pill` / `cable` / `pcb4` — noi so box
lon chi phoi bo nho chu khong phai trong so encoder. Con so 3.75x cua `carpet`
khong suy rong duoc.

VisA thi trung vi giam 2.87x, tot hon MVTec.

## VisA: lite2 TOT HON baseline

`p_ap` 110.3%, `p_f1` 103.5%. Khong phai loi do — cung tap anh, cung seed, chi
khac SAM va saliency backbone.

Gia thuyet: mask tho hon cua MobileSAM khop hon voi defect lon, mo ranh gioi.
Xu huong nay cung thay tren MVTec o muc class (`zipper` +9.4%, `capsule` +7.1%,
`cable` +5.9%), chi la tren VisA no du manh de lat ca trung binh.

Chua kiem chung. Muon khang dinh thi phai doi chieu kich thuoc defect trung binh
cua tung class voi muc thay doi `p_f1`.

## Phan bo theo class (MVTec, `p_f1`)

| Tut nhieu nhat | | Tot len | |
|---|---|---|---|
| `grid` | -13.9% | `zipper` | +9.4% |
| `wood` | -5.2% | `capsule` | +7.1% |
| `tile` | -5.0% | `cable` | +5.9% |

`grid` tut nhieu nhat dung nhu du doan cua spec muc 4.6: MobileSAM cho mask tho
hon tren defect nho.

## File

| | |
|---|---|
| `mvtec-indx-0.csv` | 15 class |
| `visa_public-indx-0.csv` | 12 class |
| `run_meta.json` | 27 entry, GPU + cau hinh + so anh moi class |

So latency chi so sanh duoc trong cung mot loai GPU. Toan bo do tren Colab T4.
