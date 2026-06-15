# SAA+ Colab Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce SAA+ paper results on MVTec and VisA (≤500 images/dataset) in a single Colab T4 notebook, reporting image-AUROC, pixel-AUROC, image-AP, pixel-AP, image-F1, pixel-F1.

**Architecture:** Three surgical changes to the existing codebase (configurable dataset paths via env vars, stratified `--max-samples` flag in eval, updated run scripts), then a new Colab notebook that downloads datasets, runs both benchmarks, and aggregates results into a summary table.

**Tech Stack:** PyTorch 2.x, GroundingDINO 0.1.0, SAM ViT-H, Python 3.12, Colab T4 (15GB VRAM), pandas for CSV aggregation.

---

## Reference: Paper Metrics

Metrics reported by `metric_cal()` in `utils/metrics.py`:

| Key | Description |
|-----|-------------|
| `i_roc` | Image-level AUROC (%) |
| `p_roc` | Pixel-level AUROC (%) |
| `i_ap` | Image-level Average Precision (%) |
| `p_ap` | Pixel-level Average Precision (%) |
| `i_f1` | Image-level max-F1 (%) |
| `p_f1` | Pixel-level max-F1 (%) |

PRO (`p_pro`) is skipped by default (`--cal-pro False`) — it adds ~30min per class.

## Dataset Sizes

- **MVTec**: 15 classes, ~5,000 test images → `--max-samples 34` → ~510 total
- **VisA**: 12 classes, ~10,000 test images → `--max-samples 42` → ~504 total

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `datasets/mvtec.py` | Modify | Make `MVTEC2D_DIR` read from `MVTEC_DIR` env var |
| `datasets/visa_public.py` | Modify | Make `VISA_DIR` read from `VISA_DIR` env var |
| `datasets/__init__.py` | Modify | Add `stratified_subset(dataset, n)` — samples proportionally from good/defect |
| `eval_SAA.py` | Modify | Add `--max-samples` arg, pass to dataloader builder |
| `run_MVTec.py` | Modify | Pass `--max-samples`, use env var for dataset root |
| `run_VisA_public.py` | Modify | Same as above |
| `demo/Benchmark_SAA.ipynb` | Create | Full Colab notebook: install → download → run → aggregate |

---

## Task 1: Make dataset paths configurable via env vars

**Files:**
- Modify: `datasets/mvtec.py:8`
- Modify: `datasets/visa_public.py:9`

- [ ] **Step 1: Update `datasets/mvtec.py`**

Replace line 8:
```python
MVTEC2D_DIR = os.environ.get('MVTEC_DIR', '../datasets/mvtec_anomaly_detection')
```

Full context after edit (lines 1-10):
```python
import glob
import os

mvtec_classes = ['carpet', 'grid', 'leather', 'tile', 'wood',
                 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill',
                 'screw', 'toothbrush', 'transistor', 'zipper']

MVTEC2D_DIR = os.environ.get('MVTEC_DIR', '../datasets/mvtec_anomaly_detection')
```

- [ ] **Step 2: Update `datasets/visa_public.py`**

Replace line 9:
```python
VISA_DIR = os.environ.get('VISA_DIR', '../datasets/VisA_pytorch/1cls')
```

Full context after edit (lines 1-12):
```python
import glob
import os
import random

visa_public_classes = ['candle', 'capsules', 'cashew', 'chewinggum',
                       'fryum', 'macaroni1', 'macaroni2',
                       'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

VISA_DIR = os.environ.get('VISA_DIR', '../datasets/VisA_pytorch/1cls')
```

- [ ] **Step 3: Verify no other files hardcode these paths**

```bash
grep -rn "mvtec_anomaly_detection\|VisA_pytorch" \
  /Users/april/code/master/vra/Segment-Any-Anomaly/ \
  --include="*.py" | grep -v __pycache__
```

Expected: only the two files just edited.

---

## Task 2: Add stratified `--max-samples` to eval pipeline

**Files:**
- Modify: `datasets/__init__.py`
- Modify: `eval_SAA.py`

- [ ] **Step 1: Add `stratified_subset` to `datasets/__init__.py`**

Add after the `denormalization` function (after line 35):

```python
def stratified_subset(dataset_inst, max_samples):
    """Return indices for a stratified subsample keeping good/defect ratio."""
    if max_samples is None or len(dataset_inst) <= max_samples:
        return list(range(len(dataset_inst)))

    labels = dataset_inst.labels
    good_idx = [i for i, l in enumerate(labels) if l == 0]
    defect_idx = [i for i, l in enumerate(labels) if l == 1]

    ratio = len(good_idx) / len(labels)
    n_good = max(1, round(max_samples * ratio))
    n_defect = max(1, max_samples - n_good)

    import random
    rng = random.Random(42)
    sampled_good = rng.sample(good_idx, min(n_good, len(good_idx)))
    sampled_defect = rng.sample(defect_idx, min(n_defect, len(defect_idx)))

    return sorted(sampled_good + sampled_defect)
```

- [ ] **Step 2: Apply subset in `get_dataloader_from_args`**

Modify `get_dataloader_from_args` to accept and apply `max_samples`:

```python
def get_dataloader_from_args(phase, **kwargs):
    dataset_inst = SAADataset(
        load_function=load_function_dict[kwargs['dataset']],
        category=kwargs['class_name'],
        phase=phase,
        k_shot=kwargs['k_shot'],
        experiment_indx=kwargs['experiment_indx']
    )

    max_samples = kwargs.get('max_samples')
    if phase == 'test' and max_samples is not None:
        from torch.utils.data import Subset
        indices = stratified_subset(dataset_inst, max_samples)
        dataset_inst = Subset(dataset_inst, indices)

    if phase == 'train':
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=True,
                                 num_workers=0)
    else:
        data_loader = DataLoader(dataset_inst, batch_size=kwargs['batch_size'], shuffle=False,
                                 num_workers=0)

    base_inst = dataset_inst.dataset if hasattr(dataset_inst, 'dataset') else dataset_inst
    debug_str = f"===> datasets: {kwargs['dataset']}, class name/len: {kwargs['class_name']}/{len(dataset_inst)}, batch size: {kwargs['batch_size']}"
    logger.info(debug_str)

    return data_loader, dataset_inst
```

- [ ] **Step 3: Add `--max-samples` arg to `eval_SAA.py`**

In `get_args()`, after the `--gpu-id` line (around line 200), add:

```python
parser.add_argument("--max-samples", type=int, default=None,
                    help="Max test images per class (stratified). None = use all.")
```

- [ ] **Step 4: Verify the arg flows into kwargs**

`eval_SAA.py` already does `kwargs = vars(args)` and passes `**kwargs` to `get_dataloader_from_args`, so `max_samples` will be picked up automatically. Confirm by grepping:

```bash
grep -n "kwargs\|max.samples" /Users/april/code/master/vra/Segment-Any-Anomaly/eval_SAA.py | head -20
```

Expected: `kwargs = vars(args)` on line ~98 and `get_dataloader_from_args(..., **kwargs)` on lines ~122/127.

---

## Task 3: Update run scripts for Colab

**Files:**
- Modify: `run_MVTec.py`
- Modify: `run_VisA_public.py`

- [ ] **Step 1: Rewrite `run_MVTec.py`**

```python
import os
from datasets import dataset_classes
from multiprocessing import Pool

if __name__ == '__main__':

    pool = Pool(processes=1)

    dataset_list = ['mvtec']
    gpu_indx = 0
    max_samples = int(os.environ.get('MAX_SAMPLES', 34))   # 34 × 15 classes ≈ 510 imgs

    for dataset in dataset_list:
        classes = dataset_classes[dataset]
        for cls in classes[:]:
            sh_method = (
                f'python eval_SAA.py '
                f'--dataset {dataset} '
                f'--class-name {cls} '
                f'--batch-size 1 '
                f'--root-dir ./result '
                f'--cal-pro False '
                f'--gpu-id {gpu_indx} '
                f'--max-samples {max_samples} '
            )
            print(sh_method)
            pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()
```

- [ ] **Step 2: Rewrite `run_VisA_public.py`**

```python
import os
from datasets import dataset_classes
from multiprocessing import Pool

if __name__ == '__main__':

    pool = Pool(processes=1)

    dataset_list = ['visa_public']
    gpu_indx = 0
    max_samples = int(os.environ.get('MAX_SAMPLES', 42))   # 42 × 12 classes ≈ 504 imgs

    for dataset in dataset_list:
        classes = dataset_classes[dataset]
        for cls in classes[:]:
            sh_method = (
                f'python eval_SAA.py '
                f'--dataset {dataset} '
                f'--class-name {cls} '
                f'--batch-size 1 '
                f'--root-dir ./result '
                f'--cal-pro False '
                f'--gpu-id {gpu_indx} '
                f'--max-samples {max_samples} '
            )
            print(sh_method)
            pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()
```

---

## Task 4: Create `demo/Benchmark_SAA.ipynb`

**File:** Create `demo/Benchmark_SAA.ipynb`

The notebook has 9 cells in order. Create the file using NotebookEdit or write the JSON directly.

### Cell 0 — markdown: title

```markdown
# SAA+ Benchmark on MVTec and VisA

Reproduces image-AUROC / pixel-AUROC / AP / F1 from the SAA+ paper  
(~500 images per dataset, stratified sampling, T4 GPU)
```

### Cell 1 — code: clone & install

```python
!git clone -b SAA-plus https://github.com/SyDuc7421/Segment-Any-Anomaly.git
%cd Segment-Any-Anomaly/

# GroundingDINO requires --no-build-isolation when torch is pre-installed
%cd GroundingDINO/
!pip install -q -e . --no-build-isolation
%cd ../SAM
!pip install -q -e .
!pip install -q "transformers>=4.30.0,<4.36.0" "supervision>=0.6.0,<0.21.0" \
    opencv-python pycocotools matplotlib onnxruntime onnx ipykernel gradio loguru
%cd ..
```

### Cell 2 — code: restart runtime

```python
# Restart so updated transformers is loaded from disk
import os
os.kill(os.getpid(), 9)
```

### Cell 3 — code: download weights

```python
%cd /content/Segment-Any-Anomaly
%mkdir -p weights
%cd weights
!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
!wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
%cd ..
```

### Cell 4 — code: download MVTec

```python
%cd /content/Segment-Any-Anomaly
%mkdir -p /content/datasets

# MVTec AD — official download (274MB per category, ~4.9GB total)
# Download via Kaggle API (requires kaggle.json in /root/.kaggle/)
# Alternative: manual upload to /content/datasets/mvtec_anomaly_detection/

# Using Kaggle:
# !pip install -q kaggle
# !mkdir -p ~/.kaggle && cp /content/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d thtuan/mvtecad -p /content/datasets/ --unzip

# Or direct wget (official MVTec mirror):
!wget -q -O /content/datasets/mvtec.tar.xz \
    "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
!tar -xf /content/datasets/mvtec.tar.xz -C /content/datasets/
!mv /content/datasets/mvtec_anomaly_detection /content/datasets/mvtec_anomaly_detection 2>/dev/null || true

import os
os.environ['MVTEC_DIR'] = '/content/datasets/mvtec_anomaly_detection'
print("MVTec classes:", os.listdir(os.environ['MVTEC_DIR']))
```

### Cell 5 — code: download VisA

```python
%cd /content/Segment-Any-Anomaly

# VisA — available on Kaggle as 'anomalib/visa'
# Requires Kaggle API key. Alternative: mount from Google Drive.

# !pip install -q kaggle
# !kaggle datasets download -d anomalib/visa -p /content/datasets/ --unzip

# Or using HuggingFace datasets (if available):
# !pip install -q datasets huggingface_hub
# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="Baiqi-Li/VisA", repo_type="dataset",
#                   local_dir="/content/datasets/visa")

# After download, set env var pointing to VisA_pytorch/1cls structure:
import os
os.environ['VISA_DIR'] = '/content/datasets/VisA_pytorch/1cls'
print("VisA classes:", os.listdir(os.environ['VISA_DIR']))
```

### Cell 6 — code: run MVTec benchmark

```python
%cd /content/Segment-Any-Anomaly
import os, subprocess

os.environ['MVTEC_DIR'] = '/content/datasets/mvtec_anomaly_detection'

# MAX_SAMPLES=34 → ~510 images across 15 classes
result = subprocess.run(
    ['python', 'run_MVTec.py'],
    env={**os.environ, 'MAX_SAMPLES': '34'},
    capture_output=False
)
print("MVTec benchmark done, exit code:", result.returncode)
```

### Cell 7 — code: run VisA benchmark

```python
%cd /content/Segment-Any-Anomaly
import os, subprocess

os.environ['VISA_DIR'] = '/content/datasets/VisA_pytorch/1cls'

# MAX_SAMPLES=42 → ~504 images across 12 classes
result = subprocess.run(
    ['python', 'run_VisA_public.py'],
    env={**os.environ, 'MAX_SAMPLES': '42'},
    capture_output=False
)
print("VisA benchmark done, exit code:", result.returncode)
```

### Cell 8 — code: aggregate and display results

```python
%cd /content/Segment-Any-Anomaly
import pandas as pd
import glob

def load_and_summarize(csv_pattern, dataset_label):
    paths = glob.glob(csv_pattern)
    if not paths:
        print(f"No CSV found for {dataset_label}: {csv_pattern}")
        return None
    df = pd.read_csv(paths[0], index_col=0)
    mean_row = df.mean(numeric_only=True).rename('MEAN')
    df = pd.concat([df, mean_row.to_frame().T])
    print(f"\n{'='*60}")
    print(f"  {dataset_label} Results (subset ~500 imgs)")
    print(f"{'='*60}")
    cols = ['i_roc', 'p_roc', 'i_ap', 'p_ap', 'i_f1', 'p_f1']
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(float_format='{:.2f}'.format))
    return df

mvtec_df = load_and_summarize('result/mvtec/*/metrics.csv', 'MVTec-AD')
visa_df   = load_and_summarize('result/visa_public/*/metrics.csv', 'VisA')
```

- [ ] **Step 1: Create the notebook file**

Create `demo/Benchmark_SAA.ipynb` as a valid Jupyter notebook JSON with the 9 cells above. Use the NotebookEdit insert workflow or write the JSON directly.

The minimal notebook JSON skeleton:
```json
{
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.12.0"}
 },
 "cells": []
}
```
Then insert each cell in order using NotebookEdit with `edit_mode: insert`.

---

## Task 5: Find correct CSV path for results aggregation

The `save_metric` function in `utils/csv_utils.py` writes to a path built by `get_dir_from_args`. Verify the exact path before the aggregation cell hardcodes it.

- [ ] **Step 1: Trace the CSV path**

```bash
grep -n "get_dir_from_args\|csv_path\|root.dir" \
    /Users/april/code/master/vra/Segment-Any-Anomaly/utils/training_utils.py | head -30
```

- [ ] **Step 2: Update Cell 8 glob pattern**

After verifying the actual path pattern from Step 1, update the `glob.glob()` calls in Cell 8 to match the real structure. Common pattern: `result/{dataset}/{experiment_name}/metrics.csv` — verify and fix.

---

## Self-Review

**Spec coverage:**
- ✅ Colab T4 compatible (notebook handles install, restart, dataset download)
- ✅ MVTec + VisA both covered (Tasks 3 + 4 in notebook)
- ✅ ~500 images/dataset via stratified `--max-samples`
- ✅ Paper metrics: i_roc, p_roc, i_ap, p_ap, i_f1, p_f1 (Task 5 aggregation)
- ✅ Pull from fork `SyDuc7421/Segment-Any-Anomaly` branch `SAA-plus`
- ⚠️ VisA download cell has two options (Kaggle / HuggingFace) with stubs — user must supply Kaggle key or adjust path. This is unavoidable since VisA requires authentication.

**Placeholder scan:** Task 4 and 5 VisA download cell contain commented alternatives — acceptable because the download method depends on the user's Kaggle credentials, which cannot be hardcoded.

**Type consistency:** `max_samples` flows as `int | None` from argparse → kwargs → `get_dataloader_from_args` → `stratified_subset`. `Subset` wraps `SAADataset`, and the loop in `eval()` calls `(data, mask, label, name, img_type)` which comes from `__getitem__` — `Subset` delegates `__getitem__` transparently. ✅
