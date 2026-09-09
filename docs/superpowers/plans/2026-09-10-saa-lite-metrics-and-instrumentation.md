# SAA-Lite Metrics & Instrumentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bổ sung toàn bộ hạ tầng đo lường và khả năng hoán đổi backbone cho SAA+ — `r_f1` (max-F1-region), latency tách giai đoạn, peak VRAM, factory SAM/saliency, và resume theo class — để mọi lần chạy GPU sau đó cho ra số liệu đầy đủ và không mất tiến độ khi Colab ngắt.

**Architecture:** Không đụng vào logic tính toán của pipeline. Bốn điểm chèn: `utils/metrics.py` (bật hàm `calculate_max_f1_region` đang bị comment), `utils/timing.py` (mới — bộ đếm giờ thuần Python, có thể test không cần GPU), `SAA/backbones.py` (mới — factory dựng SAM predictor và saliency extractor theo tên), và `run_*.py` (đọc CSV để bỏ qua class đã chạy xong). Mọi giá trị mặc định giữ nguyên hành vi cũ, nên baseline không thể bị thay đổi vô tình.

**Tech Stack:** Python 3, PyTorch, timm, scikit-learn, scikit-image, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-08-28-saa-lite-llm-prompt-design.md`

**Vị trí trong loạt plan:** Đây là **Plan 1/3**. Plan 2 (Phase A — LLM sinh prompt, mục 5 của spec) chỉ viết được sau Bước 1 của giao thức, vì nó chạy trên cấu hình Lite thắng cuộc mà Bước 1 mới chọn ra. Plan 3 (web demo FastAPI, mục 9) độc lập, viết lúc nào cũng được.

**Phạm vi plan này dừng ở đâu:** Plan này chỉ tạo ra **code và test**. Nó **không** bao gồm việc chạy benchmark (Bước 0-4 của mục 6.2 trong spec) — đó là việc tốn 18-22 giờ GPU, thực hiện sau, thủ công, trên Colab.

Hai hạng mục của spec cố ý **để lại cho Plan 2**, dù nghe cũng giống việc đo đạc: log số box còn lại sau `bbox_suppression` theo từng prompt (mục 5.8 và bảng rủi ro mục 8), và cổng quyết định về cache DINO (mục 4.5). Cả hai chỉ có ý nghĩa khi đã có prompt do LLM sinh ra để chẩn đoán, và cổng cache chỉ mở được sau khi Bước 1 cho ra số latency thật.

## Global Constraints

Các giá trị dưới đây trích nguyên văn từ spec. Mọi task đều ngầm bị ràng buộc bởi mục này.

- **Tham số khoá cứng** (spec mục 6.1) — đổi bất kỳ giá trị nào đồng nghĩa phải chạy lại toàn bộ benchmark:
  - `eval_resolution` = `400`
  - `box_threshold` = `0.1`
  - `text_threshold` = `0.1`
  - `experiment_indx` = `0` (seed `111`)
- **Baseline bit-exact** (spec mục 3, 4.1): giá trị mặc định của mọi tham số mới phải tái tạo đúng hành vi hiện tại. `sam_variant` mặc định `vit_h`, `saliency_backbone` mặc định `wide_resnet50`. Nhánh `set_property_text_prompts()` cũ **không sửa một dòng nào**.
- **Training-free** (spec mục 3, phi mục tiêu 2): không huấn luyện, không fine-tune bất kỳ mô hình nào.
- **Không refactor ngoài phạm vi** (spec mục 3, phi mục tiêu 3): chỉ chạm các file liệt kê trong mục Files của từng task.
- **Full test set** (spec mục 6.1): mọi con số đưa vào luận văn phải chạy không có `--max-samples`. Subset chỉ dùng khi profiling latency ở Bước 1.
- **`normalize()` phụ thuộc tập ảnh** (spec mục 2.7): kết quả chạy 34 ảnh và 117 ảnh **không so sánh được với nhau**. Đây là lý do tồn tại của cột `n_images` ở Task 4.
- **Quirk giữ nguyên** (spec mục 2.4): `object_prompt` parse theo vị trí từ cho ra chuỗi dính dấu phẩy (`'carpet,'`). Đây là hành vi baseline, **không sửa**.
- **Commit**: chỉ chạy lệnh `git commit` sau khi người dùng duyệt. Các bước commit dưới đây là nội dung đã soạn sẵn, không phải giấy phép tự commit.

---

## File Structure

| File | Trạng thái | Trách nhiệm |
|---|---|---|
| `tests/conftest.py` | Mới | Đưa repo root vào `sys.path` để test import được `utils`, `SAA` |
| `tests/test_metrics.py` | Mới | Test `calculate_max_f1_region` và việc `r_f1` có mặt trong `result_dict` |
| `tests/test_timing.py` | Mới | Test `StageTimer` và `summarize_timings` bằng đồng hồ giả — không cần GPU |
| `tests/test_backbones.py` | Mới | Test bảng tên hợp lệ và thông báo lỗi của factory |
| `tests/test_resume.py` | Mới | Test `completed_classes` đọc CSV đúng |
| `utils/timing.py` | Mới | `StageTimer` (context manager cộng dồn theo giai đoạn) + `summarize_timings` (gộp bản ghi từng ảnh thành các cột CSV) |
| `SAA/backbones.py` | Mới | `build_sam_predictor(variant, checkpoint, device)`, `build_saliency_extractor(name, device)` |
| `utils/metrics.py` | Sửa | Bật `r_f1` trong `metric_cal` |
| `utils/csv_utils.py` | Sửa | Thêm `completed_classes(csv_path, metric_key)` |
| `SAA/model.py` | Sửa | `__init__` nhận `sam_variant` / `saliency_backbone`; chèn mốc đo giờ vào `forward` và hai hàm con |
| `eval_SAA.py` | Sửa | Hai argument CLI mới; gộp latency/VRAM/`n_images` vào `result_dict`; ghi `run_meta.json` |
| `run_MVTec.py`, `run_VisA_public.py` | Sửa | Bỏ qua class đã có kết quả; mặc định chạy full |

Lý do tách `utils/timing.py` thành file riêng thay vì viết thẳng vào `model.py`: phần logic đo giờ không phụ thuộc torch, nên tách ra thì test được trên máy không có GPU và không có CUDA. Đây là điều kiện để Task 3 và Task 4 có test thật thay vì chỉ "chạy thử trên Colab rồi nhìn số".

---

## Task 1: Dựng bộ khung test

Repo hiện **không có thư mục `tests/`, không có pytest, không có bất kỳ test nào**. Mọi task sau đều cần nó, nên dựng trước.

Máy phát triển không cần có `torch` — các test ở Task 2, 3, 6 chỉ cần `numpy`, `scikit-learn`, `scikit-image`, `pandas`, `pytest`. Test cần torch (Task 5) sẽ tự bỏ qua khi không có torch.

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: không có.
- Produces: thư mục `tests/` chạy được bằng `pytest tests/ -v` từ repo root; mọi task sau đặt test vào đây.

- [ ] **Step 1: Tạo venv riêng cho test và cài dependency**

```bash
python3 -m venv .venv-test
.venv-test/bin/pip install --upgrade pip
.venv-test/bin/pip install pytest numpy scikit-learn scikit-image pandas
```

Venv này cố ý **không** cài `torch`. Nó chỉ dùng để chạy test logic thuần. Việc chạy pipeline thật vẫn diễn ra trên Colab với môi trường đầy đủ theo `install.sh`.

- [ ] **Step 2: Viết `tests/conftest.py`**

```python
"""Đưa repo root vào sys.path để test import được utils/ và SAA/."""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
```

- [ ] **Step 3: Viết `tests/test_smoke.py`**

```python
def test_repo_root_is_importable():
    import utils.csv_utils

    assert hasattr(utils.csv_utils, 'write_results')
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `.venv-test/bin/pytest tests/ -v`
Expected: PASS, 1 passed.

Nếu FAIL với `ModuleNotFoundError: No module named 'pandas'`, quay lại Step 1 — venv chưa cài xong.

- [ ] **Step 5: Thêm `.venv-test/` vào `.gitignore`**

Kiểm tra trước, tránh ghi trùng dòng:

```bash
grep -q '^\.venv-test/$' .gitignore || echo '.venv-test/' >> .gitignore
```

- [ ] **Step 6: Commit**

```bash
git add tests/ .gitignore
git commit -m "test: add pytest harness for metrics and timing utilities"
```

---

## Task 2: Bật `r_f1` — max-F1-region

Hàm `calculate_max_f1_region` đã được cài đầy đủ tại `utils/metrics.py:140`, nhưng **chưa bao giờ chạy**: lời gọi ở dòng 43 bị comment, nhánh `else` gán cứng `max_f1_region = 0` ở dòng 49, và `result_dict` không có key nào cho nó.

Cách hàm hoạt động, để hiểu test bên dưới: nó quét 200 mức threshold từ `score.max()` xuống `score.min()`. Ở mỗi mức, nó nhị phân hoá score map, tách connected component của cả ground-truth lẫn dự đoán bằng `measure.label(..., connectivity=2)`, rồi với **mỗi vùng dự đoán** tìm vùng ground-truth có IoU cao nhất. Vùng nào đạt IoU ≥ `pro_thresh` (0.6) tính là hit. `recall = hits / số vùng GT`, `precision = hits / số vùng dự đoán`. Kết quả trả về là F1 lớn nhất trên toàn bộ 200 mức.

**Files:**
- Modify: `utils/metrics.py:39-61`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `tests/conftest.py` từ Task 1.
- Produces: `metric_cal(scores, gt_list, gt_mask_list, cal_pro=False)` trả về dict có thêm key `'r_f1'` (float, thang 0-100). Task 4 sẽ chèn thêm key vào chính dict này.

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/test_metrics.py`:

```python
import numpy as np
import pytest

from utils.metrics import calculate_max_f1_region, metric_cal


def _one_square_mask(n_images=2, size=32, top=8, bottom=24):
    """n_images ảnh, mỗi ảnh một hình vuông đặc ở giữa."""
    masks = np.zeros((n_images, size, size), dtype=np.uint8)
    masks[:, top:bottom, top:bottom] = 1
    return masks


def test_max_f1_region_perfect_prediction_is_one():
    gt = _one_square_mask()
    scores = gt.astype(np.float64)

    assert calculate_max_f1_region(gt, scores) == pytest.approx(1.0)


def test_max_f1_region_disjoint_prediction_is_zero():
    gt = _one_square_mask(top=2, bottom=12)

    scores = np.zeros_like(gt, dtype=np.float64)
    scores[:, 20:30, 20:30] = 1.0

    assert calculate_max_f1_region(gt, scores) == pytest.approx(0.0)


def test_metric_cal_reports_r_f1_key_when_pro_disabled():
    gt_mask = _one_square_mask(n_images=4)
    gt_mask[:2] = 0
    scores = gt_mask.astype(np.float64)
    gt_list = np.array([0, 0, 1, 1])

    result = metric_cal(scores, gt_list, list(gt_mask), cal_pro=False)

    assert 'r_f1' in result
    assert result['r_f1'] == 0.0


def test_metric_cal_computes_r_f1_when_pro_enabled(monkeypatch):
    """Tách r_f1 khỏi cal_pro_metric: chỉ kiểm tra phần nối dây, không kiểm tra PRO."""
    import utils.metrics as metrics_module

    monkeypatch.setattr(metrics_module, 'cal_pro_metric', lambda *a, **k: 0.5)

    gt_mask = _one_square_mask(n_images=4)
    gt_mask[:2] = 0
    scores = gt_mask.astype(np.float64)
    gt_list = np.array([0, 0, 1, 1])

    result = metric_cal(scores, gt_list, list(gt_mask), cal_pro=True)

    assert result['r_f1'] == pytest.approx(100.0)
    assert result['p_pro'] == pytest.approx(50.0)
```

Giải thích hai test đầu, vì chúng dựa vào chi tiết cài đặt của hàm:
- **Perfect**: score map bằng đúng ground-truth. Ở bước quét đầu tiên `thred = score.max() = 1.0`, điều kiện `score > thred` không đúng với pixel nào nên không có vùng dự đoán → f1 = 0. Từ bước sau trở đi `thred < 1.0`, bản nhị phân trùng khít ground-truth → IoU = 1.0 ≥ 0.6 → recall = precision = 1 → f1 = 1. Lấy max ra 1.0.
- **Disjoint**: vùng dự đoán và vùng ground-truth không chạm nhau → IoU = 0 ở mọi mức → không có hit → f1 = 0 ở mọi mức.

Test thứ tư dùng `monkeypatch` để thay `cal_pro_metric` bằng hằng số. Có chủ đích: nó tách bạch việc "`r_f1` đã được nối vào `result_dict` chưa" khỏi việc "PRO tính đúng chưa". Hai chuyện đó hỏng vì lý do khác nhau, nên test riêng.

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `.venv-test/bin/pytest tests/test_metrics.py -v`

Expected: hai test `calculate_max_f1_region` PASS (hàm đã tồn tại sẵn), hai test `metric_cal` FAIL với `KeyError: 'r_f1'`.

Nếu hai test đầu cũng FAIL thì hàm có sẵn đang sai — dừng lại, đọc `utils/metrics.py:140-232` trước khi đi tiếp.

- [ ] **Step 3: Bật `r_f1` trong `metric_cal`**

Trong `utils/metrics.py`, thay khối `if cal_pro:` hiện tại (dòng 38-49):

```python
    # calculate max-f1 region
    if cal_pro:
        pro_auc_score = cal_pro_metric(gt_mask_list, scores, fpr_thresh=0.3)
        # calculate max-f1 region
        # max_f1_region = calculate_max_f1_region(gt_mask_list, scores)

    else:
        pro_auc_score = 0
        # pro_auc_score = 0
        # calculate max-f1 region
        max_f1_region = 0
```

thành:

```python
    # calculate max-f1 region
    if cal_pro:
        pro_auc_score = cal_pro_metric(gt_mask_list, scores, fpr_thresh=0.3)
        max_f1_region = calculate_max_f1_region(gt_mask_list, scores)
    else:
        pro_auc_score = 0
        max_f1_region = 0
```

- [ ] **Step 4: Thêm `r_f1` vào `result_dict`**

Trong cùng file, thêm một dòng vào dict trả về, ngay sau `'p_f1'`:

```python
    result_dict = {
        'i_roc': img_roc_auc * 100,
        'p_roc': per_pixel_rocauc * 100,
        'i_ap': ap_im * 100,
        'p_ap': ap_px * 100,
        'i_f1': img_f1 * 100,
        # 'i_thresh': img_threshold,
        'p_f1': pxl_f1 * 100,
        # 'p_thresh': pxl_threshold,
        'r_f1': max_f1_region * 100,
        'p_pro': pro_auc_score * 100,
    }
```

Không cần sửa `utils/csv_utils.py`: `write_results` lấy tên cột từ `results.keys()` nên cột `r_f1` tự xuất hiện trong CSV.

- [ ] **Step 5: Chạy test, xác nhận PASS**

Run: `.venv-test/bin/pytest tests/test_metrics.py -v`
Expected: PASS, 4 passed.

- [ ] **Step 6: Kiểm tra chi phí thời gian của `r_f1`**

Hàm quét 200 mức threshold, mỗi mức duyệt mọi cặp (vùng dự đoán × vùng GT) trên mọi ảnh. Chi phí tăng nhanh theo số ảnh. Cần biết con số thật trước khi bật cho một class 117 ảnh ở độ phân giải 400×400.

```bash
.venv-test/bin/python -c "
import time, numpy as np
from utils.metrics import calculate_max_f1_region
rng = np.random.default_rng(0)
gt = np.zeros((20, 400, 400), dtype=np.uint8)
gt[:, 100:150, 100:150] = 1
scores = gt.astype(np.float64) + rng.normal(0, 0.1, gt.shape)
t = time.perf_counter()
calculate_max_f1_region(gt, scores)
print(f'20 anh 400x400: {time.perf_counter() - t:.1f}s')
"
```

Ghi con số thu được vào phần Notes của plan này. Nếu 20 ảnh mất hơn 120 giây thì 117 ảnh sẽ mất hơn 10 phút mỗi class, tức khoảng 2.5 giờ CPU cho cả MVTec — lúc đó cần báo lại cho người dùng để quyết định có giảm `max_steps` xuống 100 hay không. **Không tự ý giảm `max_steps`**: đổi tham số này làm con số không so được với paper nữa.

- [ ] **Step 7: Commit**

```bash
git add utils/metrics.py tests/test_metrics.py
git commit -m "feat: report r_f1 (max-F1-region) alongside p_pro"
```

---

## Task 3: `utils/timing.py` — bộ đếm giờ tách giai đoạn

Repo hiện **không có một dòng đo thời gian nào** (`eval_SAA.py`, `run_MVTec.py`, `SAA/model.py` đều không có `time.`, `perf_counter`, hay `synchronize`).

Task này chỉ viết phần logic thuần Python, tách khỏi torch để test được không cần GPU. Task 4 mới nối nó vào pipeline.

**Điểm quan trọng nhất của task này**: CUDA chạy bất đồng bộ. Gọi `time.perf_counter()` ngay sau một lời gọi model chỉ đo được thời gian **xếp kernel vào hàng đợi**, không phải thời gian kernel chạy xong. Không có `torch.cuda.synchronize()` trước mỗi lần đọc đồng hồ thì toàn bộ bảng Pareto ở mục 4 của spec là số rác. Vì thế `sync` là tham số bắt buộc phải truyền vào, và có test riêng xác nhận nó được gọi.

**Files:**
- Create: `utils/timing.py`
- Test: `tests/test_timing.py`

**Interfaces:**
- Consumes: `tests/conftest.py` từ Task 1.
- Produces:
  - `StageTimer(clock=time.perf_counter, sync=None)` với các method: `reset()`, `stage(name)` (context manager), `snapshot()` → `dict[str, float]` đơn vị **mili-giây**.
  - `summarize_timings(records: list[dict], n_images: int, peak_vram_mb: float)` → `dict` chứa đúng các key `t_dino`, `t_sam`, `t_saliency`, `t_total`, `n_images`, `peak_vram`.
  - Task 4 dùng cả hai: `StageTimer` trong `SAA/model.py`, `summarize_timings` trong `eval_SAA.py`.

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/test_timing.py`:

```python
import pytest

from utils.timing import StageTimer, summarize_timings


class FakeClock:
    """Đồng hồ giả: trả lần lượt các mốc đã định sẵn, đơn vị giây."""

    def __init__(self, ticks):
        self.ticks = list(ticks)
        self.calls = 0

    def __call__(self):
        value = self.ticks[self.calls]
        self.calls += 1
        return value


def test_stage_records_elapsed_milliseconds():
    timer = StageTimer(clock=FakeClock([0.0, 0.5]))

    with timer.stage('dino'):
        pass

    assert timer.snapshot() == {'dino': 500.0}


def test_repeated_stage_accumulates():
    timer = StageTimer(clock=FakeClock([0.0, 0.5, 1.0, 1.25]))

    with timer.stage('dino'):
        pass
    with timer.stage('dino'):
        pass

    assert timer.snapshot() == {'dino': 750.0}


def test_sync_is_called_before_and_after_each_stage():
    calls = []

    timer = StageTimer(clock=FakeClock([0.0, 1.0]), sync=lambda: calls.append('sync'))

    with timer.stage('sam'):
        pass

    assert calls == ['sync', 'sync']


def test_sync_runs_even_when_stage_body_raises():
    calls = []
    timer = StageTimer(clock=FakeClock([0.0, 1.0]), sync=lambda: calls.append('sync'))

    with pytest.raises(ValueError):
        with timer.stage('sam'):
            raise ValueError('boom')

    assert calls == ['sync', 'sync']
    assert timer.snapshot() == {'sam': 1000.0}


def test_reset_clears_totals():
    timer = StageTimer(clock=FakeClock([0.0, 0.5]))

    with timer.stage('dino'):
        pass
    timer.reset()

    assert timer.snapshot() == {}


def test_snapshot_returns_a_copy():
    timer = StageTimer(clock=FakeClock([0.0, 0.5]))

    with timer.stage('dino'):
        pass
    snapshot = timer.snapshot()
    snapshot['dino'] = 0.0

    assert timer.snapshot() == {'dino': 500.0}


def test_summarize_timings_averages_across_images():
    records = [
        {'dino': 100.0, 'sam': 40.0, 'saliency': 10.0, 'total': 150.0},
        {'dino': 200.0, 'sam': 60.0, 'saliency': 30.0, 'total': 290.0},
    ]

    summary = summarize_timings(records, n_images=2, peak_vram_mb=1234.5)

    assert summary == {
        't_dino': 150.0,
        't_sam': 50.0,
        't_saliency': 20.0,
        't_total': 220.0,
        'n_images': 2,
        'peak_vram': 1234.5,
    }


def test_summarize_timings_handles_empty_records():
    summary = summarize_timings([], n_images=0, peak_vram_mb=0.0)

    assert summary == {
        't_dino': 0.0,
        't_sam': 0.0,
        't_saliency': 0.0,
        't_total': 0.0,
        'n_images': 0,
        'peak_vram': 0.0,
    }
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `.venv-test/bin/pytest tests/test_timing.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'utils.timing'`.

- [ ] **Step 3: Viết `utils/timing.py`**

```python
"""Đo thời gian từng giai đoạn của pipeline SAA.

Tách khỏi SAA/model.py có chủ đích: phần logic ở đây không phụ thuộc torch,
nên test được trên máy không có GPU.
"""

import time
from contextlib import contextmanager

STAGE_TO_COLUMN = (
    ('dino', 't_dino'),
    ('sam', 't_sam'),
    ('saliency', 't_saliency'),
    ('total', 't_total'),
)


class StageTimer:
    """Cộng dồn thời gian theo tên giai đoạn, đơn vị mili-giây.

    Args:
        clock: hàm trả về thời điểm hiện tại tính bằng giây.
        sync: hàm được gọi ngay trước mỗi lần đọc đồng hồ. Trên CUDA đây
            BẮT BUỘC là torch.cuda.synchronize. Kernel được xếp hàng bất
            đồng bộ, nên đọc đồng hồ mà không synchronize thì con số đo
            được là thời gian xếp hàng, không phải thời gian chạy.
    """

    def __init__(self, clock=time.perf_counter, sync=None):
        self.clock = clock
        self.sync = sync
        self.totals = {}

    def reset(self):
        self.totals = {}

    @contextmanager
    def stage(self, name):
        if self.sync is not None:
            self.sync()
        start = self.clock()
        try:
            yield
        finally:
            if self.sync is not None:
                self.sync()
            elapsed_ms = (self.clock() - start) * 1000.0
            self.totals[name] = self.totals.get(name, 0.0) + elapsed_ms

    def snapshot(self):
        return dict(self.totals)


def summarize_timings(records, n_images, peak_vram_mb):
    """Gộp danh sách snapshot theo từng ảnh thành các cột đưa vào CSV.

    Args:
        records: danh sách dict trả về từ StageTimer.snapshot(), mỗi ảnh một dict.
        n_images: số ảnh thực sự đã chạy trong lần chạy này.
        peak_vram_mb: đỉnh VRAM, đơn vị MB.
    """
    summary = {}
    divisor = len(records) if records else 1

    for stage_name, column in STAGE_TO_COLUMN:
        total = sum(record.get(stage_name, 0.0) for record in records)
        summary[column] = total / divisor if records else 0.0

    summary['n_images'] = n_images
    summary['peak_vram'] = peak_vram_mb

    return summary
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `.venv-test/bin/pytest tests/test_timing.py -v`
Expected: PASS, 8 passed.

- [ ] **Step 5: Commit**

```bash
git add utils/timing.py tests/test_timing.py
git commit -m "feat: add StageTimer for per-stage latency measurement"
```

---

## Task 4: Nối đo giờ và VRAM vào pipeline

Ba mốc đo, đặt đúng ba module mà spec mục 4.3.2 yêu cầu tách riêng:

| Giai đoạn | Vị trí chèn | Lý do đặt ở đó |
|---|---|---|
| `dino` | `SAA/model.py:208` `text_guided_region_proposal` | Hàm này được gọi một lần cho **mỗi prompt**. Đặt ở đây thì `StageTimer` tự cộng dồn cả 7 lượt DINO mỗi ảnh của class `carpet` (spec mục 2.3) |
| `sam` | `SAA/model.py:158` `set_image` và `SAA/model.py:277` `region_refine` | Hai điểm duy nhất SAM tiêu tốn GPU. `set_image` chạy một lần mỗi ảnh nhờ cờ `is_sam_set` (spec mục 2.2) |
| `saliency` | `SAA/model.py` trong `forward`, quanh `saliency_prompting` | Bao trọn phần WideResNet |
| `total` | `SAA/model.py` `forward`, bao cả thân hàm | Mẫu số của tiêu chí "nhanh hơn ≥ 3×" ở mục 7 spec |

**Files:**
- Modify: `SAA/model.py` — `__init__` (dòng 52-78), `ensemble_text_guided_mask_proposal` (dòng 157-160 và 199), `text_guided_region_proposal` (dòng 208-214), `forward` (dòng 457-500)
- Modify: `eval_SAA.py` — thân hàm `eval` (vòng lặp dòng 38-53 và chỗ dựng `result_dict` dòng 77-83)
- Test: `tests/test_timing.py` (đã có từ Task 3 — task này không thêm unit test mới, xem Step 6)

**Interfaces:**
- Consumes: `StageTimer` và `summarize_timings` từ `utils/timing.py` (Task 3); `result_dict` có key `r_f1` từ `utils/metrics.py` (Task 2).
- Produces:
  - `Model.last_timings` — `dict[str, float]`, snapshot của ảnh vừa chạy xong, cập nhật ở cuối mỗi lần `forward`.
  - `result_dict` có thêm sáu key: `t_dino`, `t_sam`, `t_saliency`, `t_total`, `n_images`, `peak_vram`.

- [ ] **Step 1: Khởi tạo timer trong `Model.__init__`**

Thêm import ở đầu `SAA/model.py`, cạnh các import sẵn có:

```python
from utils.timing import StageTimer
```

Rồi thêm vào cuối `__init__`, ngay sau `self.is_sam_set = False`:

```python
        # Đo giờ. sync là bắt buộc trên CUDA: kernel chạy bất đồng bộ nên
        # đọc đồng hồ mà không synchronize sẽ ra thời gian xếp hàng.
        self.timer = StageTimer(
            sync=torch.cuda.synchronize if 'cuda' in str(device) else None
        )
        self.last_timings = {}
```

- [ ] **Step 2: Bọc mốc `dino`**

Trong `text_guided_region_proposal` (dòng 208), đổi:

```python
    def text_guided_region_proposal(self, dino_image, object_phrase):
        # directly use the output of Grounding DINO
        boxes, logits, caption = self.get_grounding_output(
            dino_image, object_phrase, device=self.device
        )

        return boxes, logits, caption
```

thành:

```python
    def text_guided_region_proposal(self, dino_image, object_phrase):
        # directly use the output of Grounding DINO
        with self.timer.stage('dino'):
            boxes, logits, caption = self.get_grounding_output(
                dino_image, object_phrase, device=self.device
            )

        return boxes, logits, caption
```

- [ ] **Step 3: Bọc hai mốc `sam`**

Trong `ensemble_text_guided_mask_proposal`, đổi khối `set_image` (dòng 157-159):

```python
        if self.is_sam_set == False:
            self.anomaly_region_refiner.set_image(image)
            self.is_sam_set = True
```

thành:

```python
        if self.is_sam_set == False:
            with self.timer.stage('sam'):
                self.anomaly_region_refiner.set_image(image)
            self.is_sam_set = True
```

Và trong cùng hàm, đổi lời gọi `region_refine` (dòng 199):

```python
            # region 2 mask
            masks, logits = self.region_refine(ensemble_boxes, ensemble_logits, H, W)
```

thành:

```python
            # region 2 mask
            with self.timer.stage('sam'):
                masks, logits = self.region_refine(ensemble_boxes, ensemble_logits, H, W)
```

- [ ] **Step 4: Bọc mốc `saliency` và `total` trong `forward`**

Thay toàn bộ `forward` (dòng 457 tới hết) bằng:

```python
    def forward(self, image: np.ndarray):
        self.timer.reset()

        with self.timer.stage('total'):
            ####### Object TGMP for object detection
            object_masks, object_logits, object_area = self.ensemble_text_guided_mask_proposal(
                image,
                [self.object_prompt],
                ['PlaceHolder'],
                self.object_max_area,
                self.object_min_area,
                self.box_threshold,
                self.text_threshold
            )

            ###### Reasoning: set the anomaly area threshold according to object area
            self.defect_max_area = object_area * self.defect_area_threshold
            self.defect_min_area = 0.

            ####### language prompts and property prompts $\mathcal{P}^L$ $\mathcal{P}^S$
            ####### for region proposal and filter
            defect_masks, defect_logits, _ = self.ensemble_text_guided_mask_proposal(
                image,
                self.defect_prompt_list,
                self.filter_prompt_list,
                self.defect_max_area,
                self.defect_min_area,
                self.box_threshold,
                self.text_threshold
            )

            ###### saliency prompts $\mathcal{P}^S$
            with self.timer.stage('saliency'):
                defect_masks, defect_rescores, similarity_map = self.saliency_prompting(
                    image,
                    object_masks,
                    defect_masks,
                    defect_logits
                )

            ##### confidence prompts $\mathcal{P}^C$
            anomaly_map = self.confidence_prompting(defect_masks, defect_rescores, similarity_map)

            self.is_sam_set = False

        self.last_timings = self.timer.snapshot()

        appendix = {'similarity_map': similarity_map}

        return anomaly_map, appendix
```

Chỉ có hai thay đổi thật: thụt lề thân hàm vào trong `with`, và thêm ba dòng `reset` / `snapshot` / `with ... stage('saliency')`. **Không một biểu thức nào thay đổi giá trị.** Đối chiếu kỹ với bản gốc trước khi lưu — sai một dòng thụt lề ở đây là hỏng baseline.

- [ ] **Step 5: Gom số liệu trong `eval_SAA.py`**

Thêm `import torch` vào đầu `eval_SAA.py`, và `from utils.timing import summarize_timings`.

Trong hàm `eval`, thêm biến tích luỹ cạnh các list sẵn có:

```python
    names = []
    timing_records = []
```

Reset bộ đếm VRAM ngay trước vòng lặp:

```python
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
```

Trong vòng lặp, ngay sau `score, appendix = model(d)`:

```python
            score, appendix = model(d)
            timing_records.append(model.last_timings)
            scores += [score]
```

Sau khối `if/else` dựng `result_dict` (tức là sau `result_dict = metric_cal(...)`, trước `if is_vis:`), chèn:

```python
    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    result_dict.update(summarize_timings(timing_records, len(names), peak_vram_mb))
```

Đặt sau khối `if/else` chứ không đặt trong từng nhánh, vì nhánh `visa_challenge` cũng cần các cột này.

- [ ] **Step 6: Chạy lại toàn bộ test, xác nhận không hỏng gì**

Run: `.venv-test/bin/pytest tests/ -v`
Expected: PASS, tất cả test của Task 1-3 vẫn xanh.

Các thay đổi ở Step 1-5 **không có unit test riêng** — chúng cần torch, CUDA, và checkpoint model, tức là chỉ chạy được trên Colab. Nói thẳng ra thay vì giả vờ có test: phần logic đã được tách vào `utils/timing.py` và test kỹ ở Task 3; phần còn lại ở đây thuần tuý là nối dây, và được xác minh bằng Step 7.

- [ ] **Step 7: Chạy thử trên GPU (Colab) và đối chiếu số**

Trên máy có GPU và weight đầy đủ:

```bash
python eval_SAA.py --dataset mvtec --class-name carpet --max-samples 2 --cal-pro False --root-dir ./result_smoke
```

Kiểm tra ba điều trong log và trong `./result_smoke/csv/mvtec-indx-0.csv`:
1. Có đủ sáu cột mới: `t_dino`, `t_sam`, `t_saliency`, `t_total`, `n_images`, `peak_vram`.
2. `n_images` bằng đúng 2.
3. `t_dino + t_sam + t_saliency` **nhỏ hơn hoặc bằng** `t_total`. Nếu tổng ba phần lớn hơn tổng thể thì có mốc bị lồng nhau sai — quay lại Step 2-4.

Ghi lại tỉ lệ `t_dino / t_total` vào Notes. Spec mục 4.4 dự đoán DINO sẽ là nút cổ chai; đây là số đầu tiên kiểm chứng dự đoán đó.

- [ ] **Step 8: Xoá thư mục chạy thử**

```bash
rm -rf ./result_smoke
```

- [ ] **Step 9: Commit**

```bash
git add SAA/model.py eval_SAA.py
git commit -m "feat: measure per-stage latency and peak VRAM per run"
```

---

## Task 5: Factory backbone — hoán đổi SAM và saliency

Theo spec mục 2.1, SAM chỉ vào pipeline qua đúng một điểm khởi tạo và ba lời gọi (`set_image`, `transform.apply_boxes_torch`, `predict_torch`). MobileSAM và EfficientViT-SAM đều giữ nguyên interface `SamPredictor`, nên đây là drop-in thật sự — không phải viết lại `forward`.

`ModelINet` cũng đã nhận sẵn `backbone_name` và truyền thẳng cho `timm.create_model`, và phần downstream (`region_feature_extraction`) chỉ làm cosine similarity nên không hardcode số kênh. Đổi saliency backbone vì thế cũng chỉ là đổi một chuỗi.

**Files:**
- Create: `SAA/backbones.py`
- Modify: `SAA/model.py` — `__init__` (dòng 18-56)
- Modify: `eval_SAA.py` — `get_args` và chỗ dựng `SegmentAnyAnomaly.Model`
- Test: `tests/test_backbones.py`

**Interfaces:**
- Consumes: không có gì từ task trước.
- Produces:
  - `SAM_VARIANTS: dict[str, str]` — ánh xạ tên biến thể sang mô tả ngắn.
  - `SALIENCY_BACKBONES: dict[str, str]` — ánh xạ tên sang `backbone_name` của timm.
  - `build_sam_predictor(variant: str, checkpoint: str, device: str)` → đối tượng có interface `SamPredictor`.
  - `build_saliency_extractor(name: str, device: str)` → `ModelINet`.
  - `Model.__init__` nhận thêm hai keyword `sam_variant='vit_h'`, `saliency_backbone='wide_resnet50'`.
  - `eval_SAA.py` có hai flag `--sam-variant`, `--saliency-backbone`.

- [ ] **Step 1: Viết test thất bại**

Một ràng buộc phải nắm trước khi viết: `SAA/__init__.py` chỉ có hai dòng, và dòng đầu là `from .model import Model` — tức là mọi lệnh `from SAA.backbones import ...` đều kéo theo `torch`. Trên venv test (cố ý không có torch) việc đó hỏng ngay.

Cách xử lý: nạp `SAA/backbones.py` trực tiếp theo đường dẫn, bỏ qua `__init__.py`. Việc này chỉ khả thi khi `backbones.py` **không có import nặng ở cấp module** — và đó chính là ràng buộc thiết kế của Step 3. Hai chuyện khớp nhau, không phải mẹo vặt: import nặng nằm trong thân hàm còn khiến việc thiếu MobileSAM chỉ làm hỏng nhánh MobileSAM, chứ không làm hỏng cả baseline.

Tạo `tests/test_backbones.py`:

```python
import importlib.util
import os

import pytest

_BACKBONES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'SAA',
    'backbones.py',
)


def _load_backbones():
    """Nap SAA/backbones.py theo duong dan, bo qua SAA/__init__.py.

    SAA/__init__.py import .model va keo theo torch. Nap truc tiep giup
    test bang ten va duong loi chay duoc tren may khong co torch.
    """
    spec = importlib.util.spec_from_file_location('saa_backbones', _BACKBONES_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backbones = _load_backbones()


def test_module_has_no_heavy_top_level_imports():
    """Neu file nay load duoc tren venv khong co torch thi rang buoc con giu."""
    assert backbones.SAM_VARIANTS


def test_sam_variant_names_cover_the_config_grid():
    """Ba cau hinh Lite o muc 4.2 cua spec, cong baseline."""
    assert set(backbones.SAM_VARIANTS) == {'vit_h', 'mobile_sam', 'efficientvit_l0'}


def test_saliency_backbone_names_cover_the_config_grid():
    assert set(backbones.SALIENCY_BACKBONES) == {'wide_resnet50', 'mobilenetv3'}


def test_wide_resnet50_maps_to_the_timm_name_used_by_the_baseline():
    assert backbones.SALIENCY_BACKBONES['wide_resnet50'] == 'wide_resnet50_2'


def test_unknown_sam_variant_lists_the_valid_names():
    with pytest.raises(ValueError) as excinfo:
        backbones.build_sam_predictor('vit_gigantic', checkpoint='x.pth', device='cpu')

    message = str(excinfo.value)
    assert 'vit_gigantic' in message
    assert 'vit_h' in message


def test_unknown_saliency_backbone_lists_the_valid_names():
    with pytest.raises(ValueError) as excinfo:
        backbones.build_saliency_extractor('resnet9000', device='cpu')

    message = str(excinfo.value)
    assert 'resnet9000' in message
    assert 'wide_resnet50' in message
```

Sáu test này chỉ chạm phần **không cần torch**: bảng tên và đường lỗi. Việc dựng model thật cần checkpoint vài GB, không test tự động được — xác minh bằng Step 7 và Step 8.

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `.venv-test/bin/pytest tests/test_backbones.py -v`
Expected: FAIL với `FileNotFoundError` hoặc `ImportError` khi `_load_backbones()` chạy — `SAA/backbones.py` chưa tồn tại.

- [ ] **Step 3: Viết `SAA/backbones.py`**

```python
"""Factory dựng SAM predictor và saliency extractor theo tên.

Mọi import nặng đều nằm trong thân hàm, có chủ đích: thiếu MobileSAM thì
chỉ nhánh MobileSAM hỏng, nhánh baseline vit_h vẫn chạy được.
"""

SAM_VARIANTS = {
    'vit_h': 'SAM ViT-H goc, 2.4 GB — baseline',
    'mobile_sam': 'MobileSAM (vit_t)',
    'efficientvit_l0': 'EfficientViT-SAM-L0',
}

SALIENCY_BACKBONES = {
    'wide_resnet50': 'wide_resnet50_2',
    'mobilenetv3': 'mobilenetv3_large_100',
}


def build_sam_predictor(variant, checkpoint, device):
    """Trả về đối tượng có interface SamPredictor: set_image, transform, predict_torch."""
    if variant not in SAM_VARIANTS:
        raise ValueError(
            f"Unknown sam_variant '{variant}'. Valid names: {sorted(SAM_VARIANTS)}"
        )

    if variant == 'vit_h':
        from SAM.segment_anything import SamPredictor, build_sam

        return SamPredictor(build_sam(checkpoint=checkpoint).to(device))

    if variant == 'mobile_sam':
        from mobile_sam import SamPredictor, sam_model_registry

        sam = sam_model_registry['vit_t'](checkpoint=checkpoint)
        sam.to(device)
        sam.eval()
        return SamPredictor(sam)

    from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
    from efficientvit.sam_model_zoo import create_sam_model

    sam = create_sam_model(name='l0', pretrained=True, weight_url=checkpoint)
    sam = sam.to(device).eval()
    return EfficientViTSamPredictor(sam)


def build_saliency_extractor(name, device):
    """Trả về ModelINet dùng backbone tương ứng."""
    if name not in SALIENCY_BACKBONES:
        raise ValueError(
            f"Unknown saliency_backbone '{name}'. Valid names: {sorted(SALIENCY_BACKBONES)}"
        )

    from .modelinet import ModelINet

    return ModelINet(
        device=device,
        backbone_name=SALIENCY_BACKBONES[name],
        out_indices=(1, 2, 3),
    )
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `.venv-test/bin/pytest tests/test_backbones.py -v`
Expected: PASS, 6 passed.

Nếu FAIL với `ModuleNotFoundError: No module named 'torch'` thì có import nặng lọt lên cấp module — chuyển nó vào thân hàm.

- [ ] **Step 5: Nối vào `Model.__init__`**

Trong `SAA/model.py`, thêm import:

```python
from .backbones import build_sam_predictor, build_saliency_extractor
```

Đổi chữ ký `__init__`, thêm hai keyword **sau** các tham số hiện có để không phá thứ tự positional:

```python
                 ## Others
                 out_size=256,
                 device='cuda',
                 sam_variant='vit_h',
                 saliency_backbone='wide_resnet50',
```

Đổi hai dòng dựng model:

```python
        self.anomaly_region_refiner = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
```
thành
```python
        self.anomaly_region_refiner = build_sam_predictor(sam_variant, sam_checkpoint, device)
```

và

```python
        self.visual_saliency_extractor = ModelINet(device=device)
```
thành
```python
        self.visual_saliency_extractor = build_saliency_extractor(saliency_backbone, device)
```

Hai import cũ `build_sam, SamPredictor` và `ModelINet` ở đầu `SAA/model.py` giờ không còn chỗ dùng — xoá chúng đi (đây là orphan do chính thay đổi này tạo ra). Kiểm tra bằng `grep -n "SamPredictor\|build_sam\|ModelINet" SAA/model.py` trước khi xoá.

Mặc định `'vit_h'` + `'wide_resnet50'` cho ra đúng hai đối tượng như trước, nên baseline không đổi.

- [ ] **Step 6: Thêm hai flag CLI vào `eval_SAA.py`**

Trong `get_args`, thêm vào nhóm "method related parameters":

```python
    parser.add_argument('--sam-variant', type=str, default='vit_h',
                        choices=['vit_h', 'mobile_sam', 'efficientvit_l0'],
                        help='SAM backbone. vit_h la baseline.')
    parser.add_argument('--saliency-backbone', type=str, default='wide_resnet50',
                        choices=['wide_resnet50', 'mobilenetv3'],
                        help='ImageNet backbone cho saliency. wide_resnet50 la baseline.')
```

Và truyền xuống chỗ dựng model trong `main`:

```python
    model = SegmentAnyAnomaly.Model(
        dino_config_file=kwargs['dino_config_file'],
        dino_checkpoint=kwargs['dino_checkpoint'],
        sam_checkpoint=kwargs['sam_checkpoint'],
        box_threshold=kwargs['box_threshold'],
        text_threshold=kwargs['text_threshold'],
        out_size=kwargs['eval_resolution'],
        device=kwargs['device'],
        sam_variant=kwargs['sam_variant'],
        saliency_backbone=kwargs['saliency_backbone'],
    )
```

`argparse` chuyển `--sam-variant` thành khoá `sam_variant` trong `vars(args)`, nên không cần đổi tên gì thêm.

- [ ] **Step 7: Xác minh baseline không đổi trên GPU**

Đây là bước quan trọng nhất của task. Spec mục 3 yêu cầu baseline tái lập bit-exact.

```bash
python eval_SAA.py --dataset mvtec --class-name carpet --max-samples 4 --root-dir ./result_before
git stash
python eval_SAA.py --dataset mvtec --class-name carpet --max-samples 4 --root-dir ./result_after
git stash pop
diff <(cut -d, -f1-8 ./result_before/csv/mvtec-indx-0.csv) <(cut -d, -f1-8 ./result_after/csv/mvtec-indx-0.csv)
```

Expected: `diff` không in ra gì. Cắt cột 1-8 để bỏ qua các cột latency (vốn dĩ khác nhau giữa hai lần chạy) và cột `r_f1` (chưa tồn tại ở bản `stash`).

Nếu có khác biệt ở `i_roc` / `p_roc` / `i_ap` / `p_ap` / `i_f1` / `p_f1`: **dừng lại**. Factory đã làm đổi hành vi, không đi tiếp cho tới khi tìm ra nguyên nhân.

- [ ] **Step 8: Kiểm tra MobileSAM dựng được**

MobileSAM chưa có trong `install.sh`. Cài và thử:

```bash
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
wget -P weights/ https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
python eval_SAA.py --dataset mvtec --class-name carpet --max-samples 2 \
    --sam-variant mobile_sam --sam_checkpoint weights/mobile_sam.pt --root-dir ./result_mobile
```

Expected: chạy tới hết, CSV có số.

**EfficientViT-SAM là nhánh rủi ro nhất của plan này** — API `create_sam_model` / `EfficientViTSamPredictor` viết theo repo `mit-han-lab/efficientvit` và chưa được kiểm chứng trong repo này. Nếu import hoặc chữ ký hàm sai, đừng chữa cháy tại chỗ: đọc README của repo đó, sửa nhánh `efficientvit_l0` trong `SAA/backbones.py` cho khớp, rồi chạy lại Step 8 với `--sam-variant efficientvit_l0`. Nếu tốn quá một giờ, báo lại cho người dùng — spec mục 4.2 có ba cấu hình Lite và bỏ Lite-3 vẫn đủ vẽ Pareto.

- [ ] **Step 9: Xoá thư mục chạy thử**

```bash
rm -rf ./result_before ./result_after ./result_mobile
```

- [ ] **Step 10: Commit**

```bash
git add SAA/backbones.py SAA/model.py eval_SAA.py tests/test_backbones.py
git commit -m "feat: make SAM and saliency backbones selectable via CLI"
```

---

## Task 6: Resume theo class và ghi metadata mỗi lần chạy

Spec mục 6.3 gọi đây là **bắt buộc**: không có resume thì mỗi lần Colab ngắt kết nối là mất sạch tiến độ của lần full-run 3-4 giờ.

Kèm theo, spec mục 8 yêu cầu "ghi rõ số ảnh vào mọi file kết quả" để không bao giờ trộn nhầm kết quả subset với kết quả full. Cột `n_images` đã có từ Task 4; task này bổ sung file `run_meta.json` ghi thêm loại GPU — cần vì spec mục 4.3.2 nói số latency đo trên T4 và trên A100 không đặt chung bảng được.

**Files:**
- Modify: `utils/csv_utils.py`
- Modify: `run_MVTec.py`
- Modify: `run_VisA_public.py`
- Modify: `eval_SAA.py` — hàm `main`
- Test: `tests/test_resume.py`

**Interfaces:**
- Consumes: cột `n_images` trong CSV từ Task 4.
- Produces: `completed_classes(csv_path, metric_key='p_ap')` → `set[str]`.

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/test_resume.py`:

```python
import pandas as pd
import pytest

from utils.csv_utils import completed_classes


@pytest.fixture
def csv_path(tmp_path):
    return str(tmp_path / 'mvtec-indx-0.csv')


def test_missing_file_means_nothing_is_done(csv_path):
    assert completed_classes(csv_path) == set()


def test_rows_with_a_positive_metric_count_as_done(csv_path):
    pd.DataFrame(
        {'p_ap': [12.5, 0.0, 3.0]},
        index=['carpet', 'grid', 'leather'],
    ).to_csv(csv_path)

    assert completed_classes(csv_path) == {'carpet', 'leather'}


def test_placeholder_rows_are_not_done(csv_path):
    """write_results khoi tao moi class bang 0.00 truoc khi chay."""
    pd.DataFrame(
        {'p_ap': [0.0, 0.0]},
        index=['carpet', 'grid'],
    ).to_csv(csv_path)

    assert completed_classes(csv_path) == set()


def test_missing_metric_column_means_nothing_is_done(csv_path):
    pd.DataFrame({'i_roc': [90.0]}, index=['carpet']).to_csv(csv_path)

    assert completed_classes(csv_path) == set()


def test_visa_rows_keep_their_dataset_prefix(csv_path):
    """save_metric ghi 'visa_public-candle' cho dataset khac mvtec."""
    pd.DataFrame(
        {'p_ap': [5.0]},
        index=['visa_public-candle'],
    ).to_csv(csv_path)

    assert completed_classes(csv_path) == {'visa_public-candle'}
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `.venv-test/bin/pytest tests/test_resume.py -v`
Expected: FAIL với `ImportError: cannot import name 'completed_classes'`.

- [ ] **Step 3: Viết `completed_classes`**

Thêm vào cuối `utils/csv_utils.py`:

```python
def completed_classes(csv_path, metric_key='p_ap'):
    """Ten cac class da co ket qua that trong CSV.

    write_results khoi tao moi class bang 0.00 truoc khi co so, nen dieu kien
    "da xong" la metric_key > 0 chu khong phai "co dong trong file".
    """
    if not os.path.exists(csv_path):
        return set()

    df = pd.read_csv(csv_path, index_col=0)

    if metric_key not in df.columns:
        return set()

    return set(df.index[df[metric_key] > 0].astype(str))
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `.venv-test/bin/pytest tests/test_resume.py -v`
Expected: PASS, 5 passed.

- [ ] **Step 5: Nối resume vào `run_MVTec.py`**

Thay toàn bộ nội dung `run_MVTec.py` bằng:

```python
import os
from datasets import dataset_classes
from multiprocessing import Pool

from utils.csv_utils import completed_classes

if __name__ == '__main__':

    pool = Pool(processes=1)

    dataset_list = ['mvtec']
    gpu_indx = 0
    root_dir = './result'

    # Khong dat mac dinh. Spec muc 6.1: moi con so vao luan van phai chay full
    # test set, vi normalize() phu thuoc vao tap anh cua lan chay do.
    max_samples = os.environ.get('MAX_SAMPLES')

    for dataset in dataset_list:
        csv_path = os.path.join(root_dir, 'csv', f'{dataset}-indx-0.csv')
        done = completed_classes(csv_path)

        classes = dataset_classes[dataset]
        for cls in classes[:]:
            if cls in done:
                print(f'skip {cls}: da co ket qua trong {csv_path}')
                continue

            sh_method = (
                f'python eval_SAA.py '
                f'--dataset {dataset} '
                f'--class-name {cls} '
                f'--batch-size 1 '
                f'--root-dir {root_dir} '
                f'--cal-pro False '
                f'--gpu-id {gpu_indx} '
            )
            if max_samples is not None:
                sh_method += f'--max-samples {max_samples} '

            print(sh_method)
            pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()
```

Hai thay đổi so với bản cũ, cả hai đều trực tiếp từ spec: bỏ qua class đã xong (mục 6.3), và bỏ mặc định `MAX_SAMPLES=34` để full-run trở thành hành vi mặc định (mục 6.1). Muốn chạy subset để profiling thì đặt biến môi trường: `MAX_SAMPLES=4 python run_MVTec.py`.

- [ ] **Step 6: Nối resume vào `run_VisA_public.py`**

Khác `run_MVTec.py` ở một điểm dễ bỏ sót: `save_metric` trong `utils/csv_utils.py` thêm tiền tố dataset vào tên class khi dataset khác `mvtec`, nên hàng trong CSV là `visa_public-candle` chứ không phải `candle`. Khoá tra cứu phải khớp, nếu không resume sẽ không bao giờ khớp và chạy lại từ đầu mỗi lần.

Thay toàn bộ nội dung `run_VisA_public.py` bằng:

```python
import os
from datasets import dataset_classes
from multiprocessing import Pool

from utils.csv_utils import completed_classes

if __name__ == '__main__':

    pool = Pool(processes=1)

    dataset_list = ['visa_public']
    gpu_indx = 0
    root_dir = './result'

    # Khong dat mac dinh. Spec muc 6.1: moi con so vao luan van phai chay full
    # test set, vi normalize() phu thuoc vao tap anh cua lan chay do.
    max_samples = os.environ.get('MAX_SAMPLES')

    for dataset in dataset_list:
        csv_path = os.path.join(root_dir, 'csv', f'{dataset}-indx-0.csv')
        done = completed_classes(csv_path)

        classes = dataset_classes[dataset]
        for cls in classes[:]:
            # save_metric them tien to dataset cho moi dataset khac mvtec
            if f'{dataset}-{cls}' in done:
                print(f'skip {cls}: da co ket qua trong {csv_path}')
                continue

            sh_method = (
                f'python eval_SAA.py '
                f'--dataset {dataset} '
                f'--class-name {cls} '
                f'--batch-size 1 '
                f'--root-dir {root_dir} '
                f'--cal-pro False '
                f'--gpu-id {gpu_indx} '
            )
            if max_samples is not None:
                sh_method += f'--max-samples {max_samples} '

            print(sh_method)
            pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()
```

- [ ] **Step 7: Ghi `run_meta.json` trong `eval_SAA.py`**

Thêm `import json` vào đầu file. Ở cuối hàm `main`, sau lời gọi `save_metric`:

```python
    meta_path = os.path.join(os.path.dirname(csv_path), 'run_meta.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    meta[f"{kwargs['dataset']}-{kwargs['class_name']}"] = {
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'n_images': metrics.get('n_images', 0),
        'sam_variant': kwargs['sam_variant'],
        'saliency_backbone': kwargs['saliency_backbone'],
        'cal_pro': kwargs['cal_pro'],
        'eval_resolution': kwargs['eval_resolution'],
        'box_threshold': kwargs['box_threshold'],
        'text_threshold': kwargs['text_threshold'],
        'experiment_indx': kwargs['experiment_indx'],
    }

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
```

`main` cần `import os` — kiểm tra: hiện `os` chỉ được import trong khối `if __name__ == '__main__'`. Chuyển `import os` lên đầu file.

- [ ] **Step 8: Kiểm tra resume hoạt động trên GPU**

```bash
MAX_SAMPLES=2 python run_MVTec.py 2>&1 | tail -20
MAX_SAMPLES=2 python run_MVTec.py 2>&1 | grep -c '^skip'
```

Expected: lần chạy thứ hai in ra 15 dòng `skip` (đủ 15 class MVTec) và không khởi động lần chạy `eval_SAA.py` nào.

Kiểm tra `run_meta.json`:

```bash
python -c "import json; d=json.load(open('./result/csv/run_meta.json')); print(len(d), 'classes'); print(list(d.values())[0])"
```

Expected: 15 class, mỗi entry có tên GPU và `n_images: 2`.

- [ ] **Step 9: Xoá kết quả chạy thử**

Thư mục `./result` vừa tạo chứa số của subset 2 ảnh — theo spec mục 6.1 đây là số **không dùng được**, và để lại thì lần full-run sau sẽ bị resume bỏ qua toàn bộ.

```bash
rm -rf ./result
```

- [ ] **Step 10: Chạy toàn bộ test**

Run: `.venv-test/bin/pytest tests/ -v`
Expected: PASS, tất cả xanh.

- [ ] **Step 11: Commit**

```bash
git add utils/csv_utils.py run_MVTec.py run_VisA_public.py eval_SAA.py tests/test_resume.py
git commit -m "feat: resume runs per class and record run metadata"
```

---

## Sau khi xong plan này

Code đã sẵn sàng cho Bước 0 và Bước 1 của giao thức (spec mục 6.2). Việc tiếp theo là chạy GPU, không phải viết code:

1. **Bước 0** — full MVTec với `--sam-variant vit_h --cal-pro True`, đối chiếu với bảng số trong `docs/SAA+.pdf`. Nếu không khớp, dừng và debug; mọi kết quả phía sau đều vô nghĩa.
2. **Bước 1** — profiling `carpet` × 4 cấu hình. Kết quả chọn ra cấu hình Lite thắng cuộc, và đóng cổng quyết định về cache ở spec mục 4.5.
3. Sau Bước 1 mới viết được **Plan 2 (Phase A — LLM sinh prompt)**, vì Phase A chạy trên cấu hình Lite thắng cuộc.

## Notes

Điền trong lúc thực hiện:

- Thời gian `calculate_max_f1_region` trên 20 ảnh 400×400 (Task 2 Step 6): 96.6s
- Tỉ lệ `t_dino / t_total` đo trên `carpet` (Task 4 Step 7): ___
- EfficientViT-SAM dựng được không, API có phải sửa không (Task 5 Step 8): ___
