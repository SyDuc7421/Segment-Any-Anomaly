# Phase A so bo — thang so sanh nguon prompt

- **Ngay**: 2026-09-12
- **Cau hinh**: lite2 (`mobile_sam` + `mobilenetv3` + `grounding_dino`)
- **Tap anh**: full MVTec, 15 class, 1725 anh
- **GPU**: Colab T4
- **Co**: `--cal-pro False`, `--vis False`
- **Notebook**: `demo/Phase_A_Prompt_Ladder.ipynb`

Bon muc. Ba muc dau chay duoc ma **khong can LLM** - chung kiem tien de cua
Phase A truoc khi tieu GPU cho phan sinh prompt. Muc P2b la prompt do VLM cuc bo
sinh ra.

| Muc | `--prompt-source` | Prompt | Luot DINO/anh |
|---|---|---|---|
| P0 | `generic` | `"defect."` | 2 |
| P1 | `general` | 3 general_prompts | 4 |
| P2b | `llm` | Qwen2.5-VL-3B, chi biet ten class | 4.53 trung binh |
| P2v | `llm` | Qwen2.5-VL-3B + 2 anh normal 512px | 3.40 trung binh |
| P3 | `manual` | 3 general + K manual | 5.20 trung binh |

P3 lay tu lan chay Buoc 2 (`results/lite2_full/`), khong chay lai.

## Ket qua

| | `p_ap` | `p_f1` | `t_dino` | `t_total` |
|---|---|---|---|---|
| P0 | 26.39 | 35.54 | 516.8 | 735.2 |
| P1 | 26.26 | 35.80 | 1050.7 | 1504.7 |
| P2b | 23.79 | 31.16 | 1204.8 | 1527.5 |
| P2v | 21.30 | 29.32 | 871.2 | 1131.4 |
| P3 | 28.24 | **37.44** | 1394.1 | 1926.0 |

## 1. So luot DINO ty le thuan voi so prompt — xac nhan

| Muc | Luot/anh | `t_dino` | ms/luot |
|---|---|---|---|
| P0 | 2.00 | 516.7 | **258.4** |
| P1 | 4.00 | 1050.7 | **262.7** |
| P2b | 4.53 | 1204.8 | **265.8** |
| P2v | 3.40 | 871.2 | **256.2** |
| P3 | 5.20 | 1394.1 | **268.1** |

Ba con so ms/luot xap xi bang nhau tren 15 class. `t_dino` chi phu thuoc SO LUOT
goi, khong phu thuoc prompt la gi.

Don bay toc do cua Phase A la that, khong con la suy luan. Va no da duoc do sau
khi loai tru bang thuc nghiem moi duong khac (thay SAM da het du dia, thay
detector that bai — xem `results/profiling_step1/README.md`).

## 2. Texture va object di NGUOC CHIEU nhau

```
texture   P3 - P1 = -1.07 diem     prompt thu cong LAM HAI
object    P3 - P1 = +3.00 diem     prompt thu cong giup
```

Tren texture, P3 chi thang 1/5 class.

Co che ro rang: defect tren be mat van la "cho nao khac phan con lai", ma
`"defect on carpet"` dien ta dung the. Defect tren vat the la nhung thu co ten
cu the (`scratch`, `blue defect`, chan cong) ma prompt generic khong goi ra duoc.

## 3. Khong nguon nao thong tri

```
Nguon thang o tung class:   P0: 6 class    P1: 5 class    P3: 4 class
```

Bien thien theo class rat lon:

| Class | P0 | P1 | P3 | Thang | Chenh |
|---|---|---|---|---|---|
| `pill` | **63.94** | 44.76 | 44.89 | P0 | 19.18 |
| `cable` | 17.65 | 15.91 | **34.20** | P3 | 18.29 |
| `screw` | 8.53 | 20.73 | **20.81** | P3 | 12.28 |
| `leather` | 62.04 | **70.71** | 70.35 | P1 | 8.67 |
| `hazelnut` | **48.09** | 39.45 | 47.26 | P0 | 8.64 |
| `bottle` | 33.52 | **41.84** | 40.34 | P1 | 8.32 |

`pill`: mot prompt `"defect."` hon prompt thu cong **19 diem**.
`cable`: nguoc lai, prompt thu cong hon **18 diem**.

### Chi can chon dung mot trong ba la vuot chuyen gia

```
P3 (chuyen gia)              37.44
chon tot nhat moi class      39.77      +2.33 diem
```

Khong viet mot prompt moi nao. Chi chon giua ba chien luoc co san.

Cong them: 6 class thang o P0 chay nhanh **2.62x**.

## 4. P2b — LLM 3B blind: that bai ro rang

```
P2b  31.16 p_f1     thua ca P3 (37.44), P1 (35.80) va P0 (35.54)
     1.26x nhanh    nguong toi thieu 37.44 -> TRUOT
```

Ca ba muc tieu chi deu khong dat. Model: Qwen2.5-VL-3B, greedy, chi nhan ten
class. Prompt sinh ra o `SAA/prompts/generated/mvtec-blind.json`.

### Bien thien khong lo, khong phai kem deu

Thang 4/15 class:

| Class | P2b | Tot nhat trong ba nguon kia | Chenh |
|---|---|---|---|
| `capsule` | **31.23** | 21.21 | **+10.02** |
| `zipper` | **28.49** | 24.04 | +4.45 |
| `toothbrush` | **9.88** | 8.93 | +0.95 |
| `carpet` | **55.39** | 55.36 | +0.03 |

Thua 11/15, bon ca tham hoa:

| Class | P2b | Tot nhat kia | Chenh |
|---|---|---|---|
| `pill` | 11.10 | 63.94 (P0) | **-52.84** |
| `metal_nut` | 20.97 | 38.79 (P0) | -17.82 |
| `screw` | 5.86 | 20.81 (P3) | -14.95 |
| `cable` | 19.35 | 34.20 (P3) | -14.85 |

### Prompt sai te hon prompt generic

`pill` la bang chung ro nhat: mat **52.84 diem** so voi mot chu `"defect."`.

Prompt sinh cho `pill` la `['crack', 'scratch', 'stain', 'missing piece']`. Defect
`pill` trong MVTec la doi mau, in loi, nhiem ban - khong phai nut hay thieu manh.
Danh tu cu the **dan detector di sai cho** khi chung sai. `"defect."` khong ma
hoa gia dinh nao nen khong sai duoc.

Day la mot ket qua co gia tri doc lap: trong ZSAS, mot prompt cu the sai nguy
hiem hon mot prompt mo dung.

### Nhung P2b van nang tran khi la MOT LUA CHON trong tap

```
oracle chon giua 3 nguon (P0/P1/P3)   39.77
oracle chon giua 4 nguon (+P2b)       40.80      +1.03
vuot P3 (37.44)                                  +3.36
```

P2b duoc chon o `carpet`, `capsule`, `toothbrush`, `zipper`.

Nghia la **P2b te khi dung cho moi class, nhung co gia tri khi duoc chon dung
cho**. Dung luan diem trung tam: khong nguon nao thong tri.

Dang chu y: bon class P2b thang deu la nhung class ma ca ba nguon kia deu yeu.
`capsule` cao nhat trong ba nguon cu chi 21.21. Cho chuyen gia bi la cho LLM co dat.

### Gia thuyet kiem duoc

P2b thang khi defect **nhin thay duoc va co ten thong thuong** (`capsule` nut,
`zipper` lech rang). Thua tham khi defect **khong khop danh tu pho thong** -
`pill` la doi mau va in loi, `screw` la ren hong o muc vi mo.

Do chinh la thu ma **P2-vision** sinh ra de sua: cho model NHIN anh thay vi doan
tu ten class.

### Chat luong prompt, ghi lai de doi chieu voi P2-vision

- 18 cum khac nhau tren 53 prompt -> **lap lai 66%**, hoi tu ve
  `crack` / `stain` / `missing piece` / `scratch`
- 3/53 con la tinh tu tran, ke ca `rusty` tren `carpet` (tham khong gi)
- **0/53 trung khit tung chu voi prompt thu cong** -> khong co dau hieu nho du
  lieu huan luyen

## 5. P2v — cho model nhin anh: te hon nua

```
P2v  29.32 p_f1     te hon ca P2b (31.16)
     1.70x nhanh    nhanh nhat trong cac bien the LLM
```

Model duoc xem hai anh `train/good` moi class, ha xuong 512px. No xin it prompt
hon (2.4/class so voi 3.5) va lap lai it hon (47% so voi 66%) - nhung diem thap hon.

### Co che: model mo ta CAI DANG CO, con anomaly detection can CAI KHONG NEN CO

Du doan truoc khi chay la `pill` se tang, vi prompt sinh ra la `red spots` va
`letter F` - hai thu **co that** tren anh vien thuoc. Thuc te **giam**: 11.10 ->
8.08.

Ly do: chung co that tren **moi** vien thuoc, ke ca vien binh thuong. Detector
khop khap noi, phan doan thanh vo nghia.

Cung co che do giai thich `capsule`, sap 21.89 diem (31.23 -> 9.34): prompt sinh
ra la `['500', 'capsule']`. `500` la lieu luong in tren vo nang; `capsule` la
chinh vat the, bi bo loc nen loai. Class do con **0 prompt dung duoc**.

> Cho model nhin **anh binh thuong** thi no mo ta **tinh binh thuong**, roi phat
> ra chinh tinh binh thuong do lam prompt defect.

Day la mot cai gia cua viec dieu kien hoa bang thi giac, va no khong hien nhien
truoc khi do. Blind khong the mac loi nay vi no khong thay gi.

### Nhung P2v thang tuyet doi 2 class

| Class | P2b | P2v | Prompt vision |
|---|---|---|---|
| `metal_nut` | 20.97 | **40.77** | `cracked`, `deformed` |
| `toothbrush` | 9.88 | **11.79** | `cracked bristles`, `stained bristles` |

Ca hai deu cao nhat trong **ca nam nguon**. `metal_nut` voi hai tinh tu tran -
dung kieu prompt bi che o phan chat luong - vuot ca prompt thu cong cua tac gia.

### Thay doi lon nhat so voi blind

| Class | P2b | P2v | |
|---|---|---|---|
| `capsule` | 31.23 | 9.34 | **-21.89** |
| `metal_nut` | 20.97 | 40.77 | **+19.80** |
| `zipper` | 28.49 | 20.30 | -8.19 |
| `transistor` | 16.74 | 9.15 | -7.59 |

Bien thien hai chieu, khong phai kem deu.

## 6. Oracle: nam nguon, khong nguon nao thong tri

```
Nguon thang o tung class:  P1 4   P0 3   P2b 3   P3 3   P2v 2
```

Nam nguon, khong nguon nao qua 4/15.

| Chon giua | `p_f1` | Vuot P3 |
|---|---|---|
| P3 mot minh (chuyen gia) | 37.44 | — |
| oracle-3 (P0/P1/P3) | 39.77 | +2.33 |
| oracle-4 (+P2b) | 40.80 | +3.36 |
| **oracle-5 (+P2v)** | **41.06** | **+3.62** |

Ca hai bien the LLM **that bai khi dung dai tra** nhung **deu nang tran khi la
mot lua chon trong tap**. Do la ket qua trung tam cua Phase A.

## Toan bo bang theo class (`p_f1`)

| Class | P0 | P1 | P2b | P2v | P3 | Loai | Thang |
|---|---|---|---|---|---|---|---|
| carpet | 50.22 | 53.12 | **55.39** | 54.89 | 55.36 | texture | P2b |
| grid | 11.29 | **17.50** | 13.32 | 12.16 | 15.58 | texture | P1 |
| leather | 62.04 | **70.71** | 69.23 | 69.84 | 70.35 | texture | P1 |
| tile | **63.69** | 63.49 | 56.79 | 57.17 | 61.73 | texture | P0 |
| wood | 63.94 | **67.25** | 62.27 | 62.95 | 63.68 | texture | P1 |
| bottle | 33.52 | **41.84** | 29.27 | 24.79 | 40.34 | object | P1 |
| cable | 17.65 | 15.91 | 19.35 | 16.94 | **34.20** | object | P3 |
| capsule | 21.21 | 17.66 | **31.23** | 9.34 | 18.92 | object | P2b |
| hazelnut | **48.09** | 39.45 | 37.52 | 35.24 | 47.26 | object | P0 |
| metal_nut | 38.79 | 36.13 | 20.97 | **40.77** | 36.13 | object | P2v |
| pill | **63.94** | 44.76 | 11.10 | 8.08 | 44.89 | object | P0 |
| screw | 8.53 | 20.73 | 5.86 | 6.43 | **20.81** | object | P3 |
| toothbrush | 8.71 | 8.93 | 9.88 | **11.79** | 8.61 | object | P2v |
| transistor | 17.47 | 18.81 | 16.74 | 9.15 | **20.23** | object | P3 |
| zipper | 24.04 | 20.70 | **28.49** | 20.30 | 23.49 | object | P2b |

Nguon thang: P1 4 class, P0 3, P2b 3, P3 3, P2v 2. **Khong nguon nao qua 4/15.**

P1 thang P3 tren 6/15 class: `grid`, `leather`, `tile`, `wood`, `bottle`,
`toothbrush` — bon trong so do la texture.

## Mot dong can dieu tra

```
metal_nut   P1 36.13    P3 36.13
```

Bang nhau den hai chu so. `metal_nut` co prompt thu cong giau nhat —
`'blue defect. black defect. red defect. scratch.'` — va P3 = P1 cong dung
prompt do.

Bang nhau nghia la prompt ay khong dong gop mot box nao song sot vao top-k. Day
la "prompt chet" o spec muc 5.8. Bo dem `prompt_box_counts` (commit `2554aac`)
phan biet duoc hai kha nang: khong box nao qua duoc bo loc, hay co box nhung
khong lot `k_mask`.

Cell chan doan trong `demo/Phase_A_Prompt_Ladder.ipynb` chay lai mot class de
lay so nay — lan chay Buoc 2 dien ra truoc khi co bo dem.

## He qua: Phase A doi khung

Cau hoi goc — "LLM thu hep duoc bao nhieu khoang cach toi oracle?" — dua tren
gia dinh rang P3 la tran. Du lieu bac bo gia dinh do: P3 bi thua tren 6/15 class,
va bi thua boi mot chien luoc khong can chuyen gia nao viet.

| | Khung cu | Khung moi |
|---|---|---|
| Cau hoi | LLM thu hep khoang cach toi oracle? | LLM chon dung chien luoc prompt cho tung class? |
| Du dia | +1.64 diem (P1 -> P3) | **+3.62 diem tren ca P3** (oracle-5) |
| Tran | P3 | Chua biet — 41.06 chi la san |

40.80 moi chi la chon giua bon thu co san, va mot trong bon la LLM 3B blind
vua that bai khi dung dai tra. Nguon tot hon se nang tran nay len tiep.

Phat bieu cho luan van:

> Prompt thu cong cua SAA+ — thu ma paper trinh bay nhu doi hoi kien thuc chuyen
> gia — thua chien luoc generic tren 6/15 class, va lam hai co he thong tren
> texture. Chon dung nguon prompt cho tung class vuot chuyen gia 2.33 diem `p_f1`
> ma khong can viet mot prompt moi nao.

## Du lieu tho

CSV tung muc tren Google Drive: `SAA_results/phase_a_{P0,P1,P2b,P2v}/csv/mvtec-indx-0.csv`.
Prompt do LLM sinh: `SAA/prompts/generated/mvtec-{blind,vision}.json` (+ `-meta.json`).
P3 o `results/lite2_full/mvtec-indx-0.csv`.
