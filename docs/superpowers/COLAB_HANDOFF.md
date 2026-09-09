# Colab handoff — những gì Plan 1 không chạy được

- **Ngày**: 2026-09-10
- **Sinh ra từ**: `docs/superpowers/plans/2026-09-10-saa-lite-metrics-and-instrumentation.md`
- **Lý do tồn tại**: Plan 1 chạy trên máy macOS không có CUDA, không có torch, không có ảnh dataset. Mọi bước cần GPU đều bị hoãn có chủ đích, không phải bỏ sót. Đây là danh sách đầy đủ những bước đó.

---

## 0. Điều kiện tiên quyết

Notebook `demo/Benchmark_SAA.ipynb` chạy `!git clone -b dev https://github.com/SyDuc7421/Segment-Any-Anomaly.git`. Bảy commit của Plan 1 hiện **chỉ nằm local**, chưa push. Không push thì notebook kéo về bản cũ và mọi thứ dưới đây vô nghĩa.

```bash
git log --oneline 8ad9372..HEAD    # phải thấy 7 commit
git push origin dev
```

---

## 1. Bốn lỗ hổng của notebook, phải vá trước khi chạy dài

Chưa vá gì cả — dưới đây là hiện trạng và cách xử lý thủ công.

### 1.1 Resume chỉ có một chiều — nghiêm trọng nhất

`demo/Benchmark_SAA.ipynb` cell 9 copy CSV **lên** Drive, nhưng không cell nào copy **ngược về**. Colab ngắt là `/content` mất sạch; lần chạy sau `result/csv/` trống, `completed_classes` trả set rỗng, chạy lại từ class đầu tiên. Toàn bộ cơ chế resume của Task 6 thành vô dụng.

Chạy cell này **trước** cell 6 và cell 7, mỗi lần khởi động lại session:

```python
%cd /content/Segment-Any-Anomaly
from google.colab import drive
import shutil, glob, os

drive.mount('/content/drive')
src = '/content/drive/MyDrive/SAA_results'
os.makedirs('result/csv', exist_ok=True)

for p in glob.glob(f'{src}/*.csv') + glob.glob(f'{src}/run_meta.json'):
    shutil.copy(p, 'result/csv/')
    print('Restored:', os.path.basename(p))
```

Và sửa cell 9 để nó đẩy cả `run_meta.json` lên, không chỉ `*.csv`.

Kiểm tra resume có thật sự khớp: sau khi restore, chạy lại runner và đếm dòng `skip`. Nếu ra 0 trong khi CSV có số, khoá tra cứu đang sai — dừng lại, đừng để nó chạy lại 3 tiếng.

### 1.2 `--cal-pro` — đã vá, nhưng có một cái bẫy khi dùng chung với resume

**Trạng thái: đã sửa.** Trước đây `run_MVTec.py` và `run_VisA_public.py` hardcode `--cal-pro False`, mà `r_f1` (max-F1-region) nằm chung cờ đó, nên chạy qua runner thì cột `r_f1` luôn bằng 0. Đây là lỗ hổng thật của Plan 1: plan không đưa việc tham số hoá cờ này vào task nào, trong khi spec mục 6.4 yêu cầu bật `cal_pro` cho baseline, cấu hình Lite thắng cuộc, P2-vision và P3.

Cả hai runner giờ đọc biến môi trường `CAL_PRO`, mặc định `False`:

```bash
CAL_PRO=True MVTEC_DIR=/content/datasets python run_MVTec.py
```

Không đặt biến thì hành vi y hệt trước, nên baseline không đổi.

#### Bẫy: resume không biết `cal_pro` đã đổi

`completed_classes` xét `p_ap > 0`, mà `p_ap` có số ở **mọi** lần chạy, bật PRO hay không. Hệ quả:

> Chạy một class với `CAL_PRO=False` trước, rồi bật `CAL_PRO=True` chạy lại — resume sẽ **bỏ qua** class đó vì thấy `p_ap` đã có số. Cột `r_f1` và `p_pro` của class ấy ở lại `0.00` vĩnh viễn, trong khi bảng nhìn như đã hoàn tất.

Đây là loại hỏng im lặng: không báo lỗi, không thiếu dòng, chỉ thiếu số ở hai cột.

Cách tránh, chọn một:
- **Quyết định `CAL_PRO` trước khi bắt đầu một lần chạy, và không đổi giữa chừng.** Đơn giản nhất.
- Dùng `--root-dir` riêng cho lần chạy có PRO, ví dụ `./result_calpro`. Hai lần chạy hai file CSV, không giẫm lên nhau.
- Muốn bổ sung PRO cho một CSV đã có: xoá thủ công những hàng cần chạy lại (hoặc set `p_ap` của chúng về 0) trước khi chạy.

Kiểm tra nhanh trước khi tin một bảng có PRO:
```bash
python -c "
import pandas as pd
df = pd.read_csv('result/csv/mvtec-indx-0.csv', index_col=0)
missing = df.index[(df['p_ap'] > 0) & (df.get('r_f1', 0) == 0)].tolist()
print('classes co p_ap nhung thieu r_f1:', missing or 'khong co')
"
```

### 1.3 Cell tổng hợp hardcode danh sách cột

`demo/Benchmark_SAA.ipynb` cell 8:
```python
cols = [c for c in ['i_roc','p_roc','i_ap','p_ap','i_f1','p_f1'] if c in df.columns]
```
Thiếu `r_f1`, `p_pro`, và cả sáu cột mới. Số có trong CSV nhưng không hiện ra. Thay bằng:
```python
cols = [c for c in ['i_roc','p_roc','i_ap','p_ap','i_f1','p_f1','r_f1','p_pro',
                    't_dino','t_sam','t_saliency','t_total','n_images','peak_vram']
        if c in df.columns]
```

### 1.4 Chưa có cell profiling cho Bước 1

Xem mục 3 dưới đây để biết cần chạy gì.

---

## 2. Bước xác minh bị hoãn, theo từng task

Chạy theo thứ tự. Mỗi mục ghi rõ: chạy gì, kỳ vọng gì, và làm gì nếu sai.

### 2.1 Task 4 Step 7 — smoke test đo giờ

```bash
python eval_SAA.py --dataset mvtec --class-name carpet --max-samples 2 \
    --cal-pro False --root-dir ./result_smoke
```

Mở `./result_smoke/csv/mvtec-indx-0.csv`, kiểm tra ba điều:

1. Có đủ sáu cột: `t_dino`, `t_sam`, `t_saliency`, `t_total`, `n_images`, `peak_vram`.
2. `n_images` bằng đúng `2`.
3. `t_dino + t_sam + t_saliency <= t_total`.

Reviewer đã xác nhận điều 3 đúng **về mặt cấu trúc** (cả ba stage đều lồng bên trong `total`), nhưng chỉ một lần chạy thật mới chứng minh được phép cộng.

Ghi lại tỉ lệ `t_dino / t_total`. Spec mục 4.4 dự đoán DINO là nút cổ chai; đây là con số đầu tiên kiểm chứng dự đoán đó, và nó quyết định việc tối ưu SAM có còn ý nghĩa không.

Nếu điều 3 sai — tổng ba phần lớn hơn tổng thể — có mốc đo bị lồng sai. Dừng lại.

Xong thì `rm -rf ./result_smoke`.

### 2.2 Task 5 Step 7 — bit-exact baseline

**Đây là bước quan trọng nhất trong toàn bộ danh sách.** Spec mục 3 yêu cầu baseline tái lập bit-exact; cho tới khi bước này chạy, việc factory backbone không làm đổi hành vi mới chỉ được xác minh bằng đọc code, chưa bằng số.

```bash
python eval_SAA.py --dataset mvtec --class-name carpet --max-samples 4 --root-dir ./result_before
git stash
python eval_SAA.py --dataset mvtec --class-name carpet --max-samples 4 --root-dir ./result_after
git stash pop
diff <(cut -d, -f1-8 ./result_before/csv/mvtec-indx-0.csv) \
     <(cut -d, -f1-8 ./result_after/csv/mvtec-indx-0.csv)
```

Kỳ vọng: `diff` không in ra gì. Cắt cột 1-8 để bỏ qua các cột latency (khác nhau giữa hai lần chạy là bình thường) và `r_f1` (chưa tồn tại ở bản stash).

Khác biệt ở `i_roc` / `p_roc` / `i_ap` / `p_ap` / `i_f1` / `p_f1`: **dừng toàn bộ**. Không chạy benchmark nào cho tới khi tìm ra nguyên nhân.

Xong thì `rm -rf ./result_before ./result_after`.

### 2.3 Task 5 Step 8 — MobileSAM

```bash
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
wget -P weights/ https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
python eval_SAA.py --dataset mvtec --class-name carpet --max-samples 2 \
    --sam-variant mobile_sam --sam_checkpoint weights/mobile_sam.pt --root-dir ./result_mobile
```

Kỳ vọng: chạy hết, CSV có số.

### 2.4 Task 5 Step 8 — EfficientViT-SAM (nhánh rủi ro nhất)

Nhánh `efficientvit_l0` trong `SAA/backbones.py` được viết từ README của `mit-han-lab/efficientvit` và **chưa từng được thực thi**. API `create_sam_model` / `EfficientViTSamPredictor` có thể sai tên, sai chữ ký, hoặc sai đường import.

Cố ý viết trần, không bọc try/except — nhánh chưa kiểm chứng viết thẳng thì trung thực, bọc fallback là giấu rủi ro.

Nếu import hoặc chữ ký sai: đọc README của repo đó, sửa nhánh cho khớp, chạy lại. **Nếu tốn quá một giờ thì bỏ Lite-3.** Spec mục 4.2 có ba cấu hình Lite; hai cái vẫn đủ vẽ Pareto.

Xong thì `rm -rf ./result_mobile`.

### 2.5 Task 6 Step 8 — resume

```bash
MAX_SAMPLES=2 MVTEC_DIR=/content/datasets python run_MVTec.py 2>&1 | tail -20
MAX_SAMPLES=2 MVTEC_DIR=/content/datasets python run_MVTec.py 2>&1 | grep -c '^skip'
```

Kỳ vọng: lần thứ hai in ra đúng **15** dòng `skip` và không khởi động tiến trình `eval_SAA.py` nào.

Kiểm tra metadata:
```bash
python -c "import json; d=json.load(open('./result/csv/run_meta.json')); print(len(d),'classes'); print(list(d.values())[0])"
```
Kỳ vọng: 15 class, mỗi entry có tên GPU thật (không phải `'cpu'`) và `n_images: 2`.

**Rồi `rm -rf ./result` ngay.** Thư mục này chứa số của subset 2 ảnh. Để lại thì lần full-run sau sẽ bị resume bỏ qua toàn bộ 15 class — đúng cái hỏng mà Task 6 sinh ra để chặn.

---

## 3. Hai cảnh báo về chi phí

### 3.1 `r_f1` có thể rất đắt

Benchmark của Task 2: **96.6 giây cho 20 ảnh 400×400**. Ngoại suy tuyến tính theo số ảnh: ~9.4 phút cho một class 117 ảnh, ~2.4 giờ cho cả MVTec.

Nhưng con số đó **lạc quan quá mức**. Fixture đo có đúng một connected component mỗi ảnh. Anomaly map thật, khi hạ threshold xuống, vỡ thành rất nhiều mảnh nhỏ; vòng lặp trong là O(vùng dự đoán × vùng ground-truth) cho mỗi mức trong 200 mức. Chi phí thật có thể gấp nhiều lần.

**Bắt buộc**: đo `--cal-pro True` trên **một** class trước khi cam kết chạy cả 15. Nếu quá đắt, báo lại để quyết định — **không tự ý giảm `max_steps`**, vì đổi tham số đó là mất khả năng so sánh với bảng trong paper.

### 3.2 Log sẽ rất ồn

`calculate_max_f1_region` chứa sẵn hai lệnh `print()` (`utils/metrics.py:205,228`) từ trước, nằm im vì hàm chưa bao giờ được gọi. Bật `r_f1` là kích hoạt chúng: mỗi lần chạy `cal_pro=True` sẽ đổ ra hàng trăm dòng `gt_number:` / `cor recall:` mỗi class.

Không phải lỗi mới, và Plan 1 cố ý không sửa (nằm ngoài phạm vi task). Nhưng nên biết trước, và cân nhắc `2>&1 | tail` khi chạy dài.

---

## 4. Sau khi checklist này xanh hết

Theo spec mục 6.2:

- **Bước 0** — full MVTec, `--sam-variant vit_h --cal-pro True`, đối chiếu `docs/SAA+.pdf`. Không khớp thì dừng debug; mọi kết quả phía sau vô nghĩa nếu bước này sai.
- **Bước 1** — profiling `carpet` × 4 cấu hình (baseline, Lite-1, Lite-2, Lite-3), đo latency tách giai đoạn. Chọn ra cấu hình Lite thắng cuộc, và đóng cổng quyết định về cache ở spec mục 4.5.
- **Bước 2** — full run baseline + Lite trên cả MVTec và VisA.
- **Bước 3** — Phase A: sinh prompt, rồi full run P0 / P1 / P2-blind / P2-vision.
- **Bước 4** — bảng tổng hợp + web demo.

**Plan 2 (Phase A) chỉ viết được sau Bước 1**, vì nó chạy trên cấu hình Lite mà Bước 1 mới chọn ra. Plan 3 (web demo, spec mục 9) độc lập.

Notebook xuất report đã có sẵn: `report/SAA_benchmark_report.ipynb` (14 cell — bảng đầy đủ, so sánh với paper, bar chart Fp theo class, heatmap). Cell 13 hiện là ghi chú "max-F1-region chưa có"; sau khi bật `cal_pro` thì có số thật, cần cập nhật cell đó.
