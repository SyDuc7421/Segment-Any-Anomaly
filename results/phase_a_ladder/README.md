# Phase A so bo — thang so sanh nguon prompt

- **Ngay**: 2026-09-12
- **Cau hinh**: lite2 (`mobile_sam` + `mobilenetv3` + `grounding_dino`)
- **Tap anh**: full MVTec, 15 class, 1725 anh
- **GPU**: Colab T4
- **Co**: `--cal-pro False`, `--vis False`
- **Notebook**: `demo/Phase_A_Prompt_Ladder.ipynb`

Ba muc chay duoc ma **khong can LLM**. Muc dich la kiem tien de cua Phase A
truoc khi tieu GPU cho phan sinh prompt.

| Muc | `--prompt-source` | Prompt | Luot DINO/anh |
|---|---|---|---|
| P0 | `generic` | `"defect."` | 2 |
| P1 | `general` | 3 general_prompts | 4 |
| P3 | `manual` | 3 general + K manual | 5.20 trung binh |

P3 lay tu lan chay Buoc 2 (`results/lite2_full/`), khong chay lai.

## Ket qua

| | `p_ap` | `p_f1` | `t_dino` | `t_total` |
|---|---|---|---|---|
| P0 | 26.39 | 35.54 | 516.8 | 735.2 |
| P1 | 26.26 | 35.80 | 1050.7 | 1504.7 |
| P3 | 28.24 | **37.44** | 1394.1 | 1926.0 |

## 1. So luot DINO ty le thuan voi so prompt — xac nhan

| Muc | Luot/anh | `t_dino` | ms/luot |
|---|---|---|---|
| P0 | 2.00 | 516.7 | **258.4** |
| P1 | 4.00 | 1050.7 | **262.7** |
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

## Toan bo bang theo class (`p_f1`)

| Class | P0 | P1 | P3 | Loai | Thang |
|---|---|---|---|---|---|
| carpet | 50.22 | 53.12 | **55.36** | texture | P3 |
| grid | 11.29 | **17.50** | 15.58 | texture | P1 |
| leather | 62.04 | **70.71** | 70.35 | texture | P1 |
| tile | **63.69** | 63.49 | 61.73 | texture | P0 |
| wood | 63.94 | **67.25** | 63.68 | texture | P1 |
| bottle | 33.52 | **41.84** | 40.34 | object | P1 |
| cable | 17.65 | 15.91 | **34.20** | object | P3 |
| capsule | **21.21** | 17.66 | 18.92 | object | P0 |
| hazelnut | **48.09** | 39.45 | 47.26 | object | P0 |
| metal_nut | **38.79** | 36.13 | 36.13 | object | P0 |
| pill | **63.94** | 44.76 | 44.89 | object | P0 |
| screw | 8.53 | 20.73 | **20.81** | object | P3 |
| toothbrush | 8.71 | **8.93** | 8.61 | object | P1 |
| transistor | 17.47 | 18.81 | **20.23** | object | P3 |
| zipper | **24.04** | 20.70 | 23.49 | object | P0 |

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
| Du dia | +1.64 diem (P1 -> P3) | **+2.33 diem tren ca P3** |
| Tran | P3 | Chua biet — 39.77 chi la san |

39.77 moi chi la chon giua ba thu co san. LLM sinh prompt rieng co the vuot xa hon.

Phat bieu cho luan van:

> Prompt thu cong cua SAA+ — thu ma paper trinh bay nhu doi hoi kien thuc chuyen
> gia — thua chien luoc generic tren 6/15 class, va lam hai co he thong tren
> texture. Chon dung nguon prompt cho tung class vuot chuyen gia 2.33 diem `p_f1`
> ma khong can viet mot prompt moi nao.

## Du lieu tho

CSV tung muc tren Google Drive: `SAA_results/phase_a_{P0,P1}/csv/mvtec-indx-0.csv`.
P3 o `results/lite2_full/mvtec-indx-0.csv`.
