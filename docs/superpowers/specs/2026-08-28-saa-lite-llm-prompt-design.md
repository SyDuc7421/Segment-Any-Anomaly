# SAA-Lite + LLM Prompt Generation — Thiết kế cải tiến SAA+

- **Ngày**: 2026-08-28
- **Trạng thái**: Chờ review
- **Repo**: Segment-Any-Anomaly (branch `dev`)
- **Định hướng**: Đồ án nghiêng về **đóng góp nghiên cứu** (số liệu so với baseline), web demo là phụ phẩm.
- **Tài nguyên**: Colab free / Kaggle (GPU T4 16GB), có API key LLM (GPT/Claude/Gemini).

---

## 1. Bối cảnh

SAA+ (Segment Any Anomaly+) là framework phát hiện bất thường zero-shot, không cần huấn luyện. Pipeline nối ba mô hình:

1. **Grounding DINO** — sinh vùng ứng viên từ prompt ngôn ngữ
2. **SAM (ViT-H)** — tinh chỉnh box thành mask pixel-level
3. **WideResNet-50** (ImageNet pretrained) — tính saliency để chấm lại điểm

Repo hiện tại đã chạy được, có kết quả benchmark trên MVTec và VisA trong `results/`.

Đồ án này đề xuất hai cải tiến gắn liền nhau, thực hiện tuần tự:

- **Phase B — SAA-Lite**: thay các module nặng bằng module nhẹ, đo đánh đổi accuracy/latency.
- **Phase A — LLM Prompt Generation**: thay prompt viết tay thủ công bằng prompt do LLM sinh tự động.

Phase B làm trước không phải vì quan trọng hơn, mà vì nó **cắt chi phí mỗi lần chạy benchmark**, giúp các thí nghiệm của Phase A trở nên khả thi trên GPU miễn phí.

---

## 2. Hiện trạng code (đã xác minh)

Các quan sát dưới đây được kiểm chứng trực tiếp trong mã nguồn, và chúng định hình toàn bộ thiết kế.

### 2.1 SAM chỉ vào pipeline qua một điểm duy nhất

Khởi tạo tại `SAA/model.py:54`:

```python
self.anomaly_region_refiner = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
```

Và chỉ được dùng qua ba lời gọi:

| Vị trí | Lời gọi |
|---|---|
| `SAA/model.py:159` | `set_image(image)` |
| `SAA/model.py:281` | `transform.apply_boxes_torch(...)` |
| `SAA/model.py:284` | `predict_torch(...)` |

MobileSAM và EfficientViT-SAM đều giữ nguyên interface `SamPredictor`, nên việc thay thế là **drop-in thật sự** — không cần viết lại `forward`.

### 2.2 SAM encoder đã chạy một lần mỗi ảnh

Cờ `is_sam_set` tại `SAA/model.py:158-160` đảm bảo `set_image` chỉ chạy ở lần gọi `ensemble_text_guided_mask_proposal` đầu tiên; cờ được reset ở cuối `forward`. Nghĩa là phần nặng nhất (image encoder ViT-H) đã được tái sử dụng tối ưu. Thay SAM vì thế tác động thẳng vào nút cổ chai hiện tại.

### 2.3 Grounding DINO chạy nhiều lần mỗi ảnh

`Model.forward` gọi `ensemble_text_guided_mask_proposal` **hai lần**: một lần cho object (1 prompt), một lần cho defect (danh sách prompt). Bên trong, mỗi prompt là một lượt forward DINO riêng.

Số lượt DINO mỗi ảnh = `1 (object) + 3 (general_prompts) + K (manual_prompts)`.

Ví dụ class `carpet` có K=3 → **7 lượt DINO mỗi ảnh**. Đây là chi phí ẩn, và cũng là một biến có thể tối ưu.

### 2.4 Property prompt được parse theo vị trí từ

`SAA/model.py:129-134` tách tham số bằng cách đếm chỉ số từ trong câu:

```python
self.object_prompt          = property_prompts.split(' ')[7]
self.object_number          = int(property_prompts.split(' ')[5])
self.k_mask                 = int(property_prompts.split(' ')[12])
self.defect_area_threshold  = float(property_prompts.split(' ')[19])
```

Với câu mẫu trong `SAA/prompts/mvtec_parameters.py`:

```
the image of carpet have 1 dissimilar carpet, with a maximum of 5 anomaly. The anomaly would not exceed 0.9 object area.
```

chỉ số `[7]` cho ra chuỗi `'carpet,'` — **dính dấu phẩy** — và chuỗi này được dùng làm prompt DINO để dò object. Đây là quirk có sẵn; baseline giữ nguyên để không làm sai lệch so sánh.

Cách parse này quá mong manh để LLM sinh ra an toàn, nên Phase A sẽ dùng đường JSON riêng.

### 2.5 Prompt hiện tại là thủ công theo từng class

`SAA/prompts/mvtec_parameters.py` chứa từ điển viết tay:

```python
'carpet':    [['black hole', 'carpet'], ['thread', 'carpet'], ['defect.', 'carpet']],
'metal_nut': [['blue defect. black defect. red defect. scratch.', 'nut']],
```

Đây là **điểm yếu học thuật lớn nhất** của pipeline: mỗi class mới đều cần chuyên gia ngồi viết prompt, nên hệ thống không thực sự zero-shot như tên gọi. Phase A nhắm thẳng vào đây.

Điểm inject nằm ở `eval_SAA.py`, chỗ ghép:

```python
general_prompts = SegmentAnyAnomaly.build_general_prompts(kwargs['class_name'])
manual_promts   = SegmentAnyAnomaly.manul_prompts[kwargs['dataset']][kwargs['class_name']]
textual_prompts = general_prompts + manual_promts
```

### 2.6 `p_pro = 0.00` không phải bug

`run_MVTec.py:16` truyền `--cal-pro False`, và default trong `eval_SAA.py` cũng là `False`. Metric PRO đơn giản là chưa được bật. Bật lên là chuyện chi phí CPU, không phải sửa lỗi.

### 2.7 Normalize phụ thuộc vào tập ảnh — ràng buộc quan trọng nhất của giao thức

`utils/eval_utils.py:20-26`:

```python
def normalize(scores):
    max_value = np.max(scores)
    min_value = np.min(scores)
    return (scores - min_value) / (max_value - min_value)
```

Đây là min-max trên **toàn bộ score của cả class**. Hệ quả: metric phụ thuộc vào việc lần chạy đó gồm những ảnh nào. Một lần chạy 34 ảnh và một lần chạy 117 ảnh cho ra những con số **không so sánh được với nhau**.

Vì `run_MVTec.py` có `max_samples` mặc định 34, kết quả đang có trong `results/` phải được coi là **chưa phải baseline hợp lệ** cho tới khi chạy lại full.

---

## 3. Mục tiêu và phi mục tiêu

### Mục tiêu

1. Đo được đánh đổi accuracy / latency / VRAM khi thay module nặng bằng module nhẹ, ra biểu đồ Pareto.
2. Chứng minh LLM có thể thay vai chuyên gia trong việc sinh prompt, đo bằng khoảng cách tới prompt thủ công của tác giả gốc.
3. Giữ baseline tái lập được bit-exact để so sánh công bằng.
4. Một web demo chạy được, dựng trên cấu hình nhẹ.

### Phi mục tiêu (nói rõ để tránh trượt phạm vi)

1. **Không** sửa điểm yếu của SAA+ trên defect nhỏ của VisA (`candle` p_ap = 0.23). Việc đó cần tiled/multi-scale inference — đã cân nhắc và loại khỏi phạm vi. Sẽ báo cáo như giới hạn đã biết trong Future Work.
2. **Không** huấn luyện hay fine-tune bất kỳ mô hình nào. Pipeline giữ tính chất training-free.
3. **Không** refactor phần code không liên quan tới hai phase trên.
4. Web demo **không** được tính là đóng góp khoa học, chỉ là chương ứng dụng.

---

## 4. Phase B — SAA-Lite

### 4.1 Thay đổi code

| File | Thay đổi |
|---|---|
| `SAA/backbones.py` | **Mới.** Factory: `build_sam_predictor(variant, ckpt, device)` và `build_saliency_extractor(name, device)` |
| `SAA/model.py` | `Model.__init__` nhận thêm `sam_variant`, `saliency_backbone`. Mặc định `vit_h` + `wide_resnet50` |
| `eval_SAA.py` | Thêm hai argument CLI tương ứng |
| `utils/metrics.py` | Bật `r_f1` (mục 4.3.1); thêm key latency / VRAM vào `result_dict` |
| `SAA/model.py` | Chèn mốc đo thời gian quanh DINO / SAM / saliency trong `forward` (mục 4.3.2) |

Logic `forward`, bộ lọc box, saliency, rescore: **không đổi về mặt tính toán**. Phần chèn vào `forward` chỉ là đọc đồng hồ và `torch.cuda.synchronize()`, không động vào giá trị nào chảy qua pipeline. Giá trị mặc định giữ nguyên hành vi cũ, nên baseline không thể bị thay đổi vô tình.

Lưu ý về `synchronize()`: nó **chặn** cho tới khi kernel chạy xong, nên có thể làm tổng wall-clock của cả run chậm hơn đôi chút so với khi không đo. Không ảnh hưởng tới con số per-stage, nhưng nếu muốn thời gian run thật thì đo một run không instrument.

### 4.2 Lưới cấu hình

| Cấu hình | SAM | Saliency backbone |
|---|---|---|
| Baseline | SAM ViT-H (2.4 GB) | WideResNet-50 |
| Lite-1 | MobileSAM | WideResNet-50 |
| Lite-2 | MobileSAM | MobileNetV3 |
| Lite-3 | EfficientViT-SAM-L0 | WideResNet-50 |

Tách từng biến một để biết chính xác module nào gây mất accuracy, thay vì đoán.

### 4.3 Đo gì

Hai nhóm metric, **trục khác nhau, bảng khác nhau**. Nhóm accuracy so với paper; nhóm hệ thống là trục thứ hai của biểu đồ Pareto.

**Nhóm accuracy** — mỗi cấu hình ghi lại:

- Sáu metric sẵn có: `i_roc`, `p_roc`, `i_ap`, `p_ap`, `i_f1`, `p_f1`
- **`r_f1` — max-F1-region**: phải bật lên (mục 4.3.1)

`p_f1` (max-F1-pixel) và `r_f1` (max-F1-region) là **cặp metric chuẩn của ZSAS** — chính là hai cột SAA+ báo cáo. Thiếu `r_f1` thì không so được trực tiếp với bảng trong paper.

**Nhóm hệ thống**:

- **Latency tách theo giai đoạn**: DINO / SAM / saliency đo riêng, cộng thêm tổng end-to-end
- Peak VRAM

Latency tách giai đoạn là bắt buộc. Không có nó thì không biết việc tối ưu SAM còn ý nghĩa gì khi DINO chạy tới bảy lượt mỗi ảnh.

#### 4.3.1 Bật `r_f1` — hàm đã có, đang bị tắt

`utils/metrics.py:140` đã cài đầy đủ `calculate_max_f1_region(labeled_imgs, score_imgs, pro_thresh=0.6, max_steps=200)`: tách connected component bằng `measure.label`, một region tính là hit khi tỉ lệ pixel vượt ngưỡng đạt `pro_thresh`, quét 200 mức threshold rồi lấy F1 lớn nhất.

Hàm này **chưa bao giờ chạy**: lời gọi ở `metrics.py:43` bị comment, nhánh `else` gán cứng `max_f1_region = 0` (`metrics.py:49`), và `result_dict` (`metrics.py:51-61`) không có key nào cho nó. `eval_SAA.py:80` có `'r_f1': 0` nhưng chỉ ở nhánh `visa_challenge` — nhánh không tính metric.

Sửa: bỏ comment lời gọi ở nhánh `cal_pro=True`, thêm `'r_f1': max_f1_region * 100` vào `result_dict`. **Không phải sửa `utils/csv_utils.py`** — `write_results` lấy tên cột từ `results.keys()` nên cột mới tự xuất hiện.

Chi phí: cùng loại với `p_pro` — chạy CPU trên connected component, chậm. Áp cùng luật ở mục 6.4.

#### 4.3.2 Latency — điểm đo và luật đo

Hiện repo **không có một dòng đo thời gian nào** (`eval_SAA.py`, `run_MVTec.py`: không có `time.`, không có `perf_counter`). Phải thêm.

Ba mốc, bọc quanh đúng ba module trong `Model.forward`: các lượt gọi Grounding DINO, `SamPredictor`, forward saliency backbone.

**Bắt buộc `torch.cuda.synchronize()` ngay trước mỗi lần đọc đồng hồ.** CUDA chạy bất đồng bộ; không synchronize thì con số đo được là thời gian enqueue kernel, không phải thời gian chạy, và toàn bộ bảng Pareto thành rác.

Ghi theo từng ảnh rồi lấy trung bình trên class. Đưa vào `result_dict` các key `t_dino`, `t_sam`, `t_saliency`, `t_total` (ms/ảnh) và `peak_vram` (`torch.cuda.max_memory_allocated`, MB) — vào CSV theo cùng cơ chế `results.keys()` như trên.

Một cảnh báo về cách đọc số: latency phụ thuộc GPU. Mọi con số latency đưa vào luận văn phải **đo trên cùng một loại GPU**, và ghi rõ loại đó vào file kết quả. Số đo trên T4 và trên A100 không đặt chung một bảng được.

### 4.4 Giả thuyết cần kiểm chứng, không phải kết luận

Dự đoán: sau khi thay SAM, **DINO trở thành nút cổ chai mới**. Nếu đúng, Phase A tự nhiên cũng trở thành cải tiến tốc độ — LLM sinh ít prompt hơn nhưng trúng hơn sẽ vừa tăng accuracy vừa giảm số lượt DINO, khiến hai phase kể chung một câu chuyện thay vì là hai chương rời rạc.

Đây là **giả thuyết**, chỉ được khẳng định sau khi đo ở Bước 1 của giao thức.

### 4.5 Quyết định về cache — cố ý hoãn

Ý tưởng ban đầu là cache output của DINO để ablation Phase A rẻ hơn. Nhưng nếu SAA-Lite kéo được một lần chạy full MVTec xuống dưới ~1 giờ thì cache chỉ là phức tạp thừa.

**Cổng quyết định**: sau Bước 1 (profiling latency).
- Full run < ~1h → bỏ cache hoàn toàn.
- Full run > ~2h → thêm cache top-k box của DINO theo cặp `(ảnh, prompt)`, ước tính ~50 KB mỗi cặp.

### 4.6 Rủi ro

MobileSAM có thể cho mask thô hơn trên defect nhỏ của VisA, kéo `p_ap` xuống. Nếu xảy ra, đó là **kết quả cần báo cáo** — ranh giới nén được của pipeline — chứ không phải thất bại. Biểu đồ Pareto vẫn là đóng góp hợp lệ.

---

## 5. Phase A — LLM sinh prompt

### 5.1 Ý tưởng

LLM đóng vai chuyên gia, sinh ra đúng sáu thứ mà pipeline cần, xuất ra JSON, và JSON đó được cache vào repo.

### 5.2 Schema

```json
{
  "class": "carpet",
  "object_prompt": "carpet",
  "object_number": 1,
  "k_mask": 5,
  "defect_area_threshold": 0.9,
  "defect_prompts": [
    {"text": "black hole", "filter": "carpet"},
    {"text": "thread",     "filter": "carpet"}
  ]
}
```

Nhánh LLM dùng `Model.set_property_from_dict()` (thêm mới), tránh hoàn toàn trò parse theo vị trí từ ở mục 2.4. Nhánh baseline vẫn dùng `set_property_text_prompts()` cũ, **không sửa một dòng nào** — để không ai có thể nghi ngờ baseline bị làm cho tệ đi.

### 5.3 File thay đổi

| File | Thay đổi |
|---|---|
| `SAA/prompts/llm_prompts.py` | **Mới.** Loader, trả về đúng shape mà `eval_SAA.py` đang dùng |
| `SAA/prompts/generated/<dataset>.json` | **Mới.** Output của LLM, **commit vào git** |
| `tools/gen_prompts.py` | **Mới.** Script gọi LLM API sinh JSON |
| `SAA/model.py` | Thêm `set_property_from_dict()` |
| `eval_SAA.py` | Thêm `--prompt-source {manual, llm, generic}`, sửa ~4 dòng chỗ ghép prompt |

JSON sinh ra **bắt buộc phải commit**. Đó là điều kiện để hội đồng tái lập kết quả mà không cần API key.

### 5.4 Hai biến thể input

- **P2-blind** — LLM chỉ nhận tên class (ví dụ `"carpet"`). Zero-shot thuần túy, điều kiện khắt khe nhất.
- **P2-vision** — LLM nhận tên class kèm 2-3 ảnh **normal lấy từ train split**. MVTec train split gồm toàn ảnh bình thường nên vẫn hợp lệ.

### 5.5 Ranh giới rò rỉ dữ liệu

Đây là chỗ dễ mất trắng kết quả nhất khi bảo vệ.

Quy tắc bắt buộc:

- Script sinh prompt **chỉ được đọc `train` split**, hard-code trong code.
- Không chạm ảnh test.
- Không chạm ground-truth mask.
- Không chạm ảnh anomaly.

Luận văn phải ghi rõ ranh giới này kèm trích dẫn dòng code chứng minh. Nếu để lọt, toàn bộ kết quả mất giá trị.

### 5.6 Bảng so sánh chính

| Mức | Nguồn prompt | Vai trò |
|---|---|---|
| P0 | `"defect."` + property mặc định | Sàn tuyệt đối |
| P1 | `general_prompts` sẵn có trong repo | Sàn có sẵn |
| P2-blind | LLM, chỉ biết tên class | **Đóng góp** |
| P2-vision | LLM + ảnh normal | **Đóng góp** |
| P3 | `manual_prompts` của tác giả gốc | Trần (oracle) |

**P3 không phải một lần chạy riêng.** P3 chính là `manual_prompts` — tức là lần chạy Lite ở Bước 2 của giao thức. Không cần tốn thêm GPU cho nó.

Luận điểm: nếu P2 tiến sát P3, nghĩa là bỏ được chuyên gia mà vẫn giữ accuracy. Nếu P2 thua P3, kết quả vẫn công bố được, kèm phân tích class nào LLM đoán trượt và tại sao.

### 5.7 Self-refine — stretch goal, có bẫy

Ý tưởng "chấm prompt trên subset rồi cho LLM sinh lại" nghe hợp lý, nhưng **MVTec không có validation split**. Mọi subset dùng để chấm đều phải cắt từ test, tức là tự rò rỉ.

Nếu thực hiện: chia test thành val/test cố định theo seed, refine trên val, **chỉ báo cáo trên test held-out**, và nói rõ trong luận văn.

**Khuyến nghị**: bỏ, trừ khi Phase B chạy nhanh hơn dự kiến đáng kể. P2 đã đủ làm đóng góp.

### 5.8 Chi phí và rủi ro

- 15 class MVTec + 12 class VisA = 27 lời gọi mỗi biến thể, hai biến thể ≈ 54 lời gọi. Chi phí vài đô la, chạy một lần, cache vĩnh viễn.
- **Rủi ro thực tế**: LLM sinh ra cụm từ mà Grounding DINO không nắm được (từ hiếm, quá trừu tượng) → không box nào sống sót qua `bbox_suppression` → anomaly map rỗng.
- **Cách phát hiện**: log số box còn lại sau bộ lọc, theo từng prompt. Prompt nào cho 0 box trên toàn class là prompt chết. Đưa vào phần phân tích, không giấu đi.

---

## 6. Giao thức thực nghiệm

### 6.1 Luật bất di bất dịch

**Mọi cấu hình đem so sánh phải chạy trên đúng cùng một tập ảnh.** Do đặc điểm của `normalize()` ở mục 2.7, trộn kết quả subset với kết quả full là số vô nghĩa.

Chốt: **dùng full test set, không dùng `--max-samples`** cho mọi con số đưa vào luận văn. Subset chỉ được dùng khi profiling latency.

Tham số khoá cứng cho toàn bộ thí nghiệm:

| Tham số | Giá trị |
|---|---|
| `eval_resolution` | 400 |
| `box_threshold` | 0.1 |
| `text_threshold` | 0.1 |
| `experiment_indx` | 0 (seed 111) |

Đổi bất kỳ giá trị nào ở trên đồng nghĩa phải chạy lại toàn bộ.

### 6.2 Thứ tự chạy

**Bước 0 — Tái lập baseline.** Chạy full MVTec với config gốc, đối chiếu với bảng số trong `docs/SAA+.pdf`. Nếu không khớp, dừng lại debug; mọi kết quả phía sau đều vô nghĩa nếu bước này sai. Kết quả hiện có trong `results/` nghi là chạy subset nên coi như chưa có baseline.

**Bước 1 — Profiling latency.** Một class (`carpet`, 117 ảnh) × 4 cấu hình Phase B, đo latency tách giai đoạn. Tốn vài chục phút. Chọn ra cấu hình Lite thắng cuộc, và đóng cổng quyết định về cache ở mục 4.5.

**Bước 2 — Full run baseline + Lite** trên cả MVTec và VisA.

**Bước 3 — Phase A.** Sinh prompt, rồi full run P0 / P1 / P2-blind / P2-vision trên cấu hình Lite.

**Bước 4 — Bảng tổng hợp + web demo.**

### 6.3 Ngân sách GPU

Đếm cụ thể số lần full-run cần thiết:

| Lần chạy | MVTec | VisA | Ghi chú |
|---|---|---|---|
| Baseline (ViT-H, manual) | 1 | 1 | Bước 0 và Bước 2 |
| Lite (manual) = **P3** | 1 | 1 | Bước 2 |
| P0 (generic) | 1 | — | Chỉ MVTec |
| P1 (general_prompts) | 1 | — | Chỉ MVTec |
| P2-blind | 1 | 1 | |
| P2-vision | 1 | 1 | |
| **Tổng** | **6** | **4** | **10 lần full-run** |

Lite-1 / Lite-2 / Lite-3 ở mục 4.2 **không** cần full-run — chúng chỉ được profiling trên một class ở Bước 1, rồi chọn ra một cấu hình duy nhất đi tiếp.

Ước lượng thô: baseline ~3-4h mỗi run, Lite ~1.5h mỗi run → **tổng khoảng 18-22 giờ GPU T4**. Colab free chịu được nếu rải ra nhiều tuần **và có cơ chế resume**.

**Bắt buộc bổ sung resume**: `run_MVTec.py` và `run_VisA_public.py` phải đọc file CSV kết quả trước khi chạy và bỏ qua những class đã có số. `save_metric` trong `utils/csv_utils.py` đã ghi theo từng class nên việc này khả thi. Không có resume, mỗi lần Colab ngắt kết nối là mất sạch tiến độ.

Việc P0/P1 chỉ chạy trên MVTec (bỏ VisA) đã được tính sẵn vào bảng trên — tiết kiệm khoảng 3 giờ, mất mát không đáng kể vì P0/P1 chỉ đóng vai mốc sàn.

### 6.4 Metric PRO và `r_f1`

Bật `--cal-pro True` **chỉ cho các cấu hình lên bảng chính**: baseline, Lite thắng cuộc, P2-vision, P3. PRO tính trên CPU và chậm; bật cho cả 12 run là lãng phí. Các run phụ để `False`, và ghi rõ trong bảng ô nào không có số PRO.

`r_f1` đi chung cờ đó — cùng nằm trong nhánh `cal_pro` ở `metrics.py`, cùng chi phí CPU trên connected component. Hệ quả: run nào tắt cờ thì cột `r_f1` cũng trống, phải ghi rõ trong bảng.

Latency thì ngược lại — **bật cho mọi run**. Chi phí gần bằng không, và cần đủ số để vẽ Pareto.

---

## 7. Tiêu chí thành công

Chốt **trước khi chạy**. Không được chạy xong rồi mới chọn ngưỡng cho vừa với số đo được — hội đồng nhìn ra ngay.

| Phase | Đạt khi | Nếu không đạt |
|---|---|---|
| **B** | Cấu hình Lite giữ **≥ 95% `p_ap` và ≥ 95% `p_f1`** của baseline, `r_f1` không tụt quá **5 điểm tuyệt đối**, và nhanh hơn **≥ 3×** `t_total` end-to-end | Vẫn xuất bảng Pareto và kết luận "SAA+ nén được tới đâu trước khi gãy". Dùng được cho luận văn. |
| **A** | P2 vượt P1 rõ rệt trên **đa số class**, và thu hẹp **≥ 50%** khoảng cách từ P1 tới P3 | Báo cáo kết quả âm kèm phân tích class nào LLM trượt và tại sao. Vẫn là một chương hợp lệ. |

---

## 8. Bảng rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Colab ngắt giữa chừng | Resume theo từng class (mục 6.3) |
| Baseline không khớp paper | Dừng, debug trước, không đi tiếp |
| Prompt LLM chết (0 box sống sót) | Log số box sau `bbox_suppression`, đưa vào phân tích |
| MobileSAM cho mask thô trên defect nhỏ | Là kết quả, không phải lỗi — đưa vào bảng Pareto |
| Hết thời gian | Cắt theo thứ tự: self-refine → P2-vision → VisA của P0/P1 |
| Trộn nhầm kết quả subset và full | Luật ở mục 6.1; ghi rõ số ảnh vào mọi file kết quả |

---

## 9. Web demo (chương ứng dụng)

`app.py` hiện tại không deploy được: nó chứa `os.system('pip install ...')` ngay trong mã nguồn, tự tải weight, và tự `os.mkdir('weights')`.

Sau khi có SAA-Lite, demo trở nên rẻ: một service FastAPI nhận ảnh upload, chạy inference trên cấu hình nhẹ, trả về anomaly map. Tách phần cài đặt môi trường ra khỏi mã nguồn.

Xếp vào chương ứng dụng, **không** tính là đóng góp khoa học.

---

## 10. Tham khảo cho chương Related Work

**SAM phiên bản nhẹ**: MobileSAM, FastSAM, EfficientViT-SAM, EdgeSAM

**Zero-shot / few-shot anomaly detection**: WinCLIP, APRIL-GAN (winner VAND 2023), AnomalyCLIP (ICLR 2024), AdaCLIP, ClipSAM

**LLM/VLM kết hợp anomaly detection**: AnomalyGPT, Myriad, GPT-4V-AD, Customizing VLM for AD

**Open-vocabulary detection**: YOLO-World, OWLv2

---

## 11. Tổng hợp file bị ảnh hưởng

| File | Loại |
|---|---|
| `SAA/backbones.py` | Mới |
| `SAA/prompts/llm_prompts.py` | Mới |
| `SAA/prompts/generated/*.json` | Mới (commit vào git) |
| `tools/gen_prompts.py` | Mới |
| `SAA/model.py` | Sửa: thêm tham số backbone, thêm `set_property_from_dict()` |
| `eval_SAA.py` | Sửa: thêm CLI args, đổi chỗ ghép prompt |
| `run_MVTec.py`, `run_VisA_public.py` | Sửa: thêm resume, bỏ mặc định `max_samples` |
| `app.py` | Viết lại thành service (giai đoạn cuối) |

Không file nào khác được đụng tới.
