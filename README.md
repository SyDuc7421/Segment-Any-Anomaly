# Segment Any Anomaly
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12Sh0j92YYmTa0oIuSEWWpPBCpIwCSVhz?usp=sharing)
[![HuggingFace Space](https://img.shields.io/badge/🤗-HuggingFace%20Space-cyan.svg)](https://huggingface.co/spaces/Caoyunkang/Segment-Any-Anomaly)

📢 **Latest News**  
🌟 Our paper has been officially accepted by **IEEE Transactions on Cybernetics**! Check out the [formal publication](https://ieeexplore.ieee.org/document/10884560).

This repository contains the official implementation of [Segment Any Anomaly without Training via Hybrid Prompt Regularization, SAA+](http://arxiv.org/abs/2305.10724).

SAA+ aims to segment any anomaly without the need for training. We achieve this by adapting existing foundation models, 
namely [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and 
[Segment Anything](https://github.com/facebookresearch/segment-anything), with hybrid prompt regularization.

---

# Fork: SAA-Lite

> Phan nay la **cong viec cua fork**, khong phai cua tac gia goc. Toan bo README
> ban goc nam nguyen ben duoi.

Do xem SAA+ nen duoc toi dau truoc khi gay: thay tung module nang bang module
nhe, va do danh doi accuracy / toc do / VRAM tren full test set.

## Ket qua chinh

Cau hinh **lite2** = MobileSAM + MobileNetV3, giu nguyen Grounding DINO.
Full test set, Colab T4.

| | MVTec (15 class, 1725 anh) | | VisA (12 class, 2162 anh) | |
|---|---|---|---|---|
| | baseline | lite2 | baseline | lite2 |
| max-F1-pixel | 37.72 | **37.44** (99.3%) | 33.74 | **34.92** (103.5%) |
| `p_ap` | 28.86 | 28.24 (97.8%) | 22.07 | 24.35 (110.3%) |
| ms/anh | 3994 | **1926** (2.07x) | 4756 | **3110** (1.53x) |
| Full run | 1.91h | 0.92h | 2.86h | 1.87h |

Accuracy giu duoc gan nhu nguyen ven; tren VisA con vuot baseline. Toc do gap
doi, nhung khong dat moc 3x — tran ly thuyet cua truc SAM chi la 2.28x.

## Bon phat hien

**1. SAM khong phai nut co chai, Grounding DINO moi la.** Thay SAM ViT-H bang
MobileSAM lam tang toc tang do 23x (2295 -> 99 ms) nhung end-to-end chi 2.07x,
vi DINO khong doi. Sau khi thay, ty trong DINO tang tu 34% len 72%.

**2. Grounding DINO khong thay the duoc.** Thu YOLO-World va OWLv2: ca hai vua
kem chinh xac hon (con 4-16% `p_ap`) vua **cham hon**. Chung duoc huan luyen de
do danh tu vat the, khong do cum mo ta khuyet tat nhu `"black hole"`, `"thread"`.

**3. Cai dat max-F1-region trong repo goc khong khop dinh nghia ma paper phat
bieu, va co the tra ve gia tri > 1** (class `wood` ra 122.57). Tai hien duoc
bang mot vi du 5 dong. Fork nay bao cao ca hai cot: `r_f1` (ban goc, de so voi
bang da cong bo) va `r_f1_fixed` (dung dinh nghia).

**4. Chon cau hinh bang mot class la rui ro.** Profiling tren rieng `carpet` cho
93.0% `p_ap` — truot nguong 95%. Tren ca 15 class la 97.8% — dat. Ba ket luan
rut ra tu mot class deu lech khi kiem lai tren full test set.

## Bo sung vao ma nguon

| | |
|---|---|
| `r_f1`, `r_f1_fixed` | max-F1-region: bat ham co san nhung bi tat, va them ban cai dung dinh nghia |
| `t_dino`, `t_sam`, `t_saliency`, `t_total` | Latency tach theo giai doan, co `torch.cuda.synchronize` truoc moi lan doc dong ho |
| `peak_vram`, `n_images` | Ghi kem moi ket qua |
| `--sam-variant`, `--saliency-backbone`, `--detector` | Hoan doi backbone tu dong lenh |
| Resume theo class | Doi chieu `run_meta.json` truoc khi bo qua, nen ket qua chay subset khong bi nham la full |
| `tests/` | 64 test (repo goc khong co test nao) |

Mac dinh cua moi tham so moi tai tao dung hanh vi cu, nen baseline giu bit-exact.

## Tai lieu

| | |
|---|---|
| Bao cao tong hop | [`docs/report-phase-b.md`](docs/report-phase-b.md) |
| Baseline full-run | [`results/baseline_full/`](results/baseline_full/) |
| Profiling Buoc 1 | [`results/profiling_step1/`](results/profiling_step1/) |
| Full-run lite2 | [`results/lite2_full/`](results/lite2_full/) |
| Thiet ke | [`docs/superpowers/specs/`](docs/superpowers/specs/) |

Notebook Colab: `demo/Benchmark_SAA.ipynb` (baseline), `demo/Profiling_SAA_Lite.ipynb`
(Buoc 1), `demo/Benchmark_SAA_Lite2.ipynb` (Buoc 2).

## Hai cau hoi con mo

- **VisA cao hon paper 24.6%** o lan chay baseline. Pipeline deterministic nen
  day khong phai nhieu. Chua dua so VisA vao ket luan tai lap cho toi khi tra
  loi duoc — nghi split hoac subset khac.
- **lite2 tot hon baseline tren VisA.** Gia thuyet: mask tho hon cua MobileSAM
  khop hon voi defect lon, mo ranh gioi. Chua kiem chung.

## Tiep theo

Phase A — cho LLM sinh prompt thay cho tu dien viet tay theo tung class. Sau
Phase B thi no khong con chi la chuyen accuracy: DINO chiem 72% va so luot goi
DINO ty le thuan voi so prompt, nen **giam so prompt la don bay toc do duy nhat
con lai**.

---

## :fire:What's New

- We have added a [Huggingface demo](https://huggingface.co/spaces/Caoyunkang/Segment-Any-Anomaly). Enjoy it~
- We have updated the [colab demo](https://colab.research.google.com/drive/12Sh0j92YYmTa0oIuSEWWpPBCpIwCSVhz?usp=sharing). Enjoy it~
- We have updated this repository for SAA+.
- We have published [Segment Any Anomaly without Training via Hybrid Prompt Regularization, SAA+](http://arxiv.org/abs/2305.10724).


## :gem:Framework
We found that a simple assembly of foundation models suffers from severe language ambiguity. 
Therefore, we introduce hybrid prompts derived from domain expert knowledge and target image context to alleviate the language ambiguity. 
The framework is illustrated below:

![Framework](./assets/framework.png)

## Quick Start

### :bank:Dataset Preparation

We evaluate SAA+ on four public datasets: MVTec-AD, VisA, KSDD2, and MTD. 
Additionally, SAA+ was a winning team in the [VAND workshop](https://sites.google.com/view/vand-cvpr23/challenge), 
which offers a specified dataset, VisA-Challenge. To prepare the datasets, please follow the instructions below:

By default, we save the data in the `../datasets` directory.

```bash
cd $ProjectRoot # e.g., /home/SAA
cd ..
mkdir datasets
cd datasets
```

Then, follow the corresponding instructions to prepare individual datasets:

- [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
- [VisA-Public](https://github.com/search?q=spot+the+difference&type=repositories)
- [VisA-Challenge](https://codalab.lisn.upsaclay.fr/competitions/12499)
- [KSDD2](https://www.vicos.si/resources/kolektorsdd2/)
- [MTD](https://github.com/abin24/Magnetic-tile-defect-datasets.)

### :hammer:Environment Setup
You can use our script for one-click setup of the environment and downloading the checkpoints.

```bash
cd $ProjectRoot
bash install.sh
```

### :page_facing_up:Repeat the public results

**MVTec-AD**

```bash
python run_MVTec.py
```

**VisA-Public**

```bash
python run_VisA_public.py
```

**VisA-Challenge**

```bash
python run_VAND_workshop.py
```

The submission files can be found in `./result_VAND_workshop/visa_challenge-k-0/0shot`.

**KSDD2**

```bash
python run_KSDD2.py
```

**MTD**

```bash
python run_MTD.py
```

### :page_facing_up:Demo Results

Run following command for demo results

```bash
python demo.py
```

![Demo](./assets/demo_result.png)

## :dart:Performance
![Results](./assets/results.png)
![Qualitative Results](./assets/qualitative_results.png)


## :hammer: Todo List

We have planned the following features to be added in the near future:

- [x] Update repository for SAA+
- [X] Detail the zero-shot anomaly detection framework.
- [x] Evaluate on other image anomaly detection datasets.
- [x] Add UI for easy evaluation.
- [x] Update Colab demo.
- [x] HuggingFace demo.

## 💘 Acknowledgements
Our work is largely inspired by the following projects. Thanks for their admiring contribution.

- [WinClip](https://github.com/caoyunkang/WinClip)
- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Grounded Segment Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)


## Stargazers over time

[![Stargazers over time](https://starchart.cc/caoyunkang/Segment-Any-Anomaly.svg)](https://starchart.cc/caoyunkang/Segment-Any-Anomaly)


## Citation

If you find this project helpful for your research, please consider citing the following BibTeX entry.

```BibTex

@article{cao_segment_2023,
	title = {Segment Any Anomaly without Training via Hybrid Prompt Regularization},
	url = {http://arxiv.org/abs/2305.10724},
	number = {{arXiv}:2305.10724},
	publisher = {{arXiv}},
	author = {Cao, Yunkang and Xu, Xiaohao and Sun, Chen and Cheng, Yuqi and Du, Zongwei and Gao, Liang and Shen, Weiming},
	urldate = {2023-05-19},
	date = {2023-05-18},
	langid = {english},
	eprinttype = {arxiv},
	eprint = {2305.10724 [cs]},
	keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Artificial Intelligence},
}

@article{kirillov2023segany,
  title={Segment Anything}, 
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

@inproceedings{ShilongLiu2023GroundingDM,
  title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
  author={Shilong Liu and Zhaoyang Zeng and Tianhe Ren and Feng Li and Hao Zhang and Jie Yang and Chunyuan Li and Jianwei Yang and Hang Su and Jun Zhu and Lei Zhang},
  year={2023}
}
```
