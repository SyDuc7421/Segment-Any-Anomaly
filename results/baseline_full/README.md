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

**VisA cao hon paper 24.6%.** Pipeline deterministic (seed co dinh, khong huan
luyen), nen chenh lech khong the do nhieu.

### Dieu tra 2026-09-13: bon gia thuyet bi loai

**Khong phai do split.** `datasets/visa_public.py:9` mac dinh
`VisA_pytorch/1cls`, dung split chuan. So anh test doi chieu voi ky vong cua
1cls (10% normal + toan bo 100 anomalous):

| Class | Do duoc | Ky vong |
|---|---|---|
| candle, macaroni1, macaroni2, pcb1, pcb2 | 200 | 200 |
| pcb3, pcb4 | 201 | 200 (lam tron) |
| capsules | 160 | 160 |
| cashew, chewinggum, fryum, pipe_fryum | 150 | 150 |

11/12 khop tuyet doi.

**Khong phai do prompt bi sua.** `git log SAA/prompts/visa_parameters.py` chi co
mot commit: `638746e SAA+`, lan import goc. Fork chua dung toi.

**Khong phai lech he thong.** Chenh lech tap trung o **2/12 class**:
`chewinggum` 86.12 va `capsules` 59.27, trong khi muoi class kia nam trong
khoang 8.93 den 52.52. Bo hai class cao nhat: mean con **25.95** so voi 27.07
cua paper.

**Khong phai rieng mot metric, khong phai do resize.** `r_f1` lech cung huong
(MVTec thap hon, VisA cao hon). `specify_resolution` (`utils/eval_utils.py:5`)
resize score va mask giong het nhau cho ca hai dataset.

### Gia thuyet con lai, chua kiem duoc

Prompt trong repo co the **khac** prompt da tao ra bang so trong paper. Paper
viet o muc 5.1: *"Details about the prompts derived from domain expert knowledge
are explained in the supplementary material"* - va supplementary khong co trong
repo. Khong co gi bao dam `visa_parameters.py` da cong bo trung voi thu chay ra
Table 1.

Doi chieu per-class cung khong lam duoc: paper chi cong bo trung binh moi dataset.

### Cach viet cho luan van

Bao cao so do duoc, neu ro da loai tru split / prompt bi sua / resize / metric,
va ghi rang chenh lech tap trung o hai class. **Khong tuyen bo "tai lap thanh
cong tren VisA"**, va cung khong lang tranh con so.

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
