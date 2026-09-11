# Baseline full-run — SAA+ goc

- **Ngay**: 2026-09-12
- **Cau hinh**: `sam_variant=vit_h`, `saliency_backbone=wide_resnet50`, prompt thu cong cua tac gia
- **Tap anh**: full test set, khong dung `--max-samples`
  - MVTec: 15 class, 1725 anh
  - VisA (public, 1cls): 12 class, 2162 anh
- **Tham so khoa cung** (spec muc 6.1): `eval_resolution=400`, `box_threshold=0.1`, `text_threshold=0.1`, `experiment_indx=0` (seed 111)
- **`cal_pro`**: bat, nen co `p_pro`, `r_f1`, `r_f1_fixed`

## Doi chieu voi paper

| | Do duoc | Paper | Lech |
|---|---|---|---|
| MVTec `p_f1` | 37.72 | 39.40 | -4.3% |
| MVTec `r_f1` | 42.45 | 49.67 | -14.5% |
| VisA `p_f1` | 33.74 | 27.07 | **+24.6%** |
| VisA `r_f1` | 15.95 | 14.46 | +10.3% |

`p_f1` tren MVTec lech -4.3%, coi nhu tai lap duoc.

**VisA cao hon paper 24.6% - chua giai thich duoc.** Pipeline deterministic (seed
co dinh, khong huan luyen), nen chenh lech khong the do nhieu. Mot dataset thap
4%, dataset kia cao 25%, nguoc chieu nhau - dau hieu khac biet he thong. Gia
thuyet chua loai tru: paper dung split VisA khac (`2cls`/`4cls` thay vi `1cls`),
hoac dung `visa_challenge` thay vi `visa_public`. Can dieu tra truoc khi dua so
VisA vao luan van.

## Hai cot max-F1-region

- `r_f1` - ban goc cua SAA+. Dung cot nay de so voi 49.67, vi nhieu kha nang
  paper tinh bang chinh doan code do. **Co the vuot 100**: class `wood` ra
  122.57. Recall dem so vung du doan khop duoc roi chia cho so vung GT, nen
  nhieu manh du doan cung trum mot vung GT la recall > 1.
- `r_f1_fixed` - cai dung dinh nghia ma chinh paper phat bieu (TP lay tu phep
  ghep mot-mot giua vung du doan va vung GT). Luon <= 100. **Day moi la so
  dung**, nhung khong so ngang voi bang da cong bo duoc.

Ban cu thoi phong khoang 2x tren MVTec (42.45 so voi 21.92). Class `zipper` co
`r_f1` = 10.91 trong khi `r_f1_fixed` = 0.00 - khong vung du doan nao dat IoU
0.6 voi vung GT nao, toan bo 10.91 la ao.

## Latency

Do tren Colab T4. So latency do tren GPU khac **khong dat chung bang duoc**.

| | MVTec | VisA |
|---|---|---|
| `t_total` | 3994 ms/anh | 4756 ms/anh |
| `t_dino / t_total` | 0.338 | 0.404 |
| `peak_vram` | 8.8 GB | 9.0 GB |

DINO chiem 1/3 thoi gian va **khong doi khi thay SAM**, nen tran tang toc ly
thuyet cua Phase B chi khoang 2.25x - duoi moc >=3x o spec muc 7.

Uoc full MVTec ~1.9h, duoi nguong 2h nen theo spec muc 4.5: **bo cache DINO**.

## Thieu gi

Khong co `run_meta.json` di kem. Lan chay nay dien ra truoc khi notebook ghi
thang vao Drive, nen metadata khong duoc giu lai. Hai hau qua:

1. Khong co ban ghi doc lap ve loai GPU va gia tri `max_samples` cua lan chay.
   Bang tren dua vao cot `n_images` trong CSV de ket luan day la full run.
2. Copy CSV nay vao `result/csv/` de resume se **khong** duoc tin: runner thay
   thieu metadata se chay lai tat ca.
