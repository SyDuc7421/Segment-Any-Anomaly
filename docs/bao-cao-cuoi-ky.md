# BÁO CÁO CUỐI KỲ

> **Hướng dẫn dùng file này**
>
> File tổ chức theo đúng ba phần yêu cầu. Chỗ `[ĐIỀN]` là thông tin chỉ nhóm
> biết. Phần III cần ghép slide đã báo cáo giữa kỳ — file này chỉ đưa dàn ý và
> những chỗ cần cập nhật theo kết quả mới.
>
> Xuất PDF: `pandoc docs/bao-cao-cuoi-ky.md -o bao-cao.pdf` hoặc mở bằng Typora /
> VS Code rồi Export PDF. Ghép với slide PDF bằng `pdfunite` hoặc bất kỳ công cụ
> merge nào. Tên file nộp: `MSHV1-MSHV2-MSHV3.pdf`.

---

# PHẦN I — BÁO CÁO TÓM TẮT

## 1. Thông tin nhóm

| Họ và tên | MSHV | Đóng góp |
|---|---|---|
| [ĐIỀN] | [ĐIỀN] | [ĐIỀN] |
| [ĐIỀN] | [ĐIỀN] | [ĐIỀN] |
| [ĐIỀN] | [ĐIỀN] | [ĐIỀN] |

## 2. Tên đề tài

**SAA-Lite: Phân tích và nén Segment Any Anomaly+ cho bài toán phân đoạn bất
thường zero-shot**

Gợi ý tên tiếng Anh: *SAA-Lite: Compressing and Auditing Segment Any Anomaly+
for Zero-Shot Anomaly Segmentation*

## 3. Tóm tắt nội dung đã nghiên cứu

Đề tài lấy SAA+ (IEEE Trans. Cybernetics 2025) làm điểm xuất phát. SAA+ là
framework phân đoạn bất thường **không cần huấn luyện**, nối ba mô hình nền:
Grounding DINO sinh vùng ứng viên từ prompt ngôn ngữ, SAM tinh chỉnh box thành
mask pixel, và WideResNet-50 tính saliency để chấm lại điểm.

Nhóm thực hiện ba việc:

**(a) Tái lập baseline** trên MVTec-AD và VisA với full test set, xây dựng hạ
tầng đo lường mà repo gốc không có: latency tách theo giai đoạn, peak VRAM, cơ
chế resume theo class, và bộ test tự động.

**(b) Nén pipeline (Phase B)** — thay từng module nặng bằng module nhẹ, đo đánh
đổi accuracy / tốc độ / bộ nhớ, dựng biểu đồ Pareto.

**(c) Phân tích nguồn prompt (Phase A)** — so sánh năm chiến lược sinh prompt,
từ generic đến prompt thủ công của tác giả đến prompt do LLM sinh tự động.

## 4. Nội dung đã mở rộng ngoài phạm vi ban đầu

| Mở rộng | Lý do |
|---|---|
| Thay Grounding DINO bằng YOLO-World và OWLv2 | Sau khi đo thấy DINO chiếm 72% thời gian, nó là đòn bẩy tốc độ duy nhất còn lại |
| Cài lại metric max-F1-region | Phát hiện cài đặt gốc trả về giá trị > 1, không khớp định nghĩa paper |
| Sinh prompt bằng VLM trọng số mở chạy cục bộ | Không phụ thuộc API trả phí; tái lập được 100% nhờ greedy decoding trên trọng số cố định |
| Phân tích oracle chọn nguồn prompt theo từng class | Dữ liệu cho thấy không nguồn nào thống trị |

## 5. Kết quả chính

### 5.1 Nén pipeline

Cấu hình **lite2** (MobileSAM + MobileNetV3, giữ Grounding DINO), full test set:

| | MVTec baseline | MVTec lite2 | Giữ được |
|---|---|---|---|
| max-F1-pixel | 37.72 | 37.44 | **99.3%** |
| `p_ap` | 28.86 | 28.24 | 97.8% |
| ms/ảnh | 3994 | **1926** | **2.07× nhanh** |

Trên VisA, lite2 còn **vượt** baseline (110.3% `p_ap`).

### 5.2 Bảy phát hiện

Bốn phát hiện nhắm thẳng vào tiền đề của paper gốc:

1. **Nút cổ chai là Grounding DINO, không phải SAM.** Thay SAM ViT-H bằng
   MobileSAM làm tầng đó nhanh 23× nhưng end-to-end chỉ 2.07×. Tỉ trọng DINO
   tăng từ 34% lên 72%. Ngược trực giác khi nhìn kích thước model (636M so với
   172M tham số).

2. **Grounding DINO không thay thế được.** YOLO-World và OWLv2 đều vừa kém chính
   xác hơn (còn 4-16% `p_ap`) vừa **chậm hơn**. Không có điểm Pareto nào.

3. **Prompt thủ công theo từng class — thứ paper trình bày như đòi hỏi kiến thức
   chuyên gia — thua chiến lược generic trên 6/15 class**, và làm hại có hệ thống
   trên class texture (−1.07 điểm) trong khi giúp trên object (+3.00 điểm).

4. **Cài đặt max-F1-region của repo gốc không khớp định nghĩa paper phát biểu và
   có thể trả về giá trị > 1** (class `wood` cho 122.57). Tái hiện được bằng một
   ví dụ 5 dòng.

Hai phát hiện về bản chất prompt trong ZSAS:

5. **Prompt cụ thể SAI nguy hiểm hơn prompt mơ hồ ĐÚNG.** Class `pill` mất
   **52.84 điểm** khi prompt là `crack`/`missing piece` thay vì một chữ
   `"defect."` — vì defect của `pill` trong MVTec là đổi màu và in lỗi.

6. **Cho model nhìn ảnh bình thường khiến nó mô tả tính bình thường làm prompt
   defect.** Class `capsule` sinh ra prompt `500` — liều lượng in trên vỏ nang —
   và sập 21.89 điểm.

Một bài học phương pháp:

7. **Chọn cấu hình bằng một class là rủi ro.** Cả ba kết luận rút ra từ `carpet`
   đều lệch khi kiểm lại trên 15 class.

### 5.3 Không nguồn prompt nào thống trị

| Nguồn | max-F1-pixel | Nhanh hơn P3 |
|---|---|---|
| P0 `"defect."` | 35.54 | **2.62×** |
| P1 `general_prompts` | 35.80 | 1.28× |
| P2b LLM, chỉ biết tên class | 31.16 | 1.26× |
| P2v LLM + ảnh normal | 29.32 | 1.70× |
| P3 prompt thủ công | **37.44** | 1.00× |
| **chọn tốt nhất mỗi class** | **41.06** | — |

Năm nguồn, **không nguồn nào thắng quá 4/15 class**. Chọn đúng nguồn cho từng
class vượt chuyên gia **3.62 điểm** mà không viết một prompt mới nào.

LLM 3B thất bại ở cả hai biến thể khi dùng đại trà, nhưng **cả hai đều nâng trần
khi là một lựa chọn trong tập**.

---

# PHẦN II — BÁO CÁO CHI TIẾT

## 1. Bối cảnh và pipeline gốc

SAA+ nối ba mô hình:

```
Grounding DINO  →  tìm cái gì, ở đâu      (ngôn ngữ)
SAM             →  vẽ chính xác đường bao (hình học)
WideResNet-50   →  cái này có thật lạ không (thị giác)
```

Ba nguồn tin bổ sung nhau. Bỏ saliency thì DINO bắt nhầm tràn lan; bỏ SAM thì chỉ
có box chứ không có mask.

Số lượt gọi DINO mỗi ảnh = `1 (object) + 3 (general) + K (manual)`. Đây là chi phí
ẩn, và cũng là một biến tối ưu được.

## 2. Thiết lập thí nghiệm

| Tham số | Giá trị |
|---|---|
| `eval_resolution` | 400 |
| `box_threshold`, `text_threshold` | 0.1 |
| `experiment_indx` | 0 (seed 111) |
| Tập ảnh | Full test set, không dùng `--max-samples` |
| Phần cứng | Colab T4 16 GB |

**Ràng buộc quan trọng nhất.** Hàm chuẩn hoá điểm là min-max trên **toàn bộ score
của cả class**:

```python
# utils/eval_utils.py
def normalize(scores):
    max_value = np.max(scores)
    min_value = np.min(scores)
    value_range = max_value - min_value
    if value_range == 0:
        return np.zeros_like(scores, dtype=float)
    return (scores - min_value) / value_range
```

Hệ quả: kết quả chạy 34 ảnh và 117 ảnh **không so sánh được với nhau**. Mọi con số
trong báo cáo này đều từ full test set.

*(Nhánh `value_range == 0` là do nhóm thêm — xem mục 7.)*

## 3. Hạ tầng đo lường đã xây dựng

Repo gốc không có test nào và không đo thời gian. Nhóm bổ sung:

### 3.1 Đo latency tách giai đoạn

```python
# utils/timing.py
class StageTimer:
    """Cộng dồn thời gian theo tên giai đoạn, đơn vị mili-giây.

    sync: hàm được gọi ngay trước mỗi lần đọc đồng hồ. Trên CUDA đây BẮT BUỘC là
    torch.cuda.synchronize. Kernel được xếp hàng bất đồng bộ, nên đọc đồng hồ mà
    không synchronize thì con số đo được là thời gian xếp hàng, không phải thời
    gian chạy.
    """

    def __init__(self, clock=time.perf_counter, sync=None):
        self.clock = clock
        self.sync = sync
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
```

Điểm mấu chốt là `sync`. Không có nó thì toàn bộ bảng Pareto là số rác nhưng
nhìn vẫn hợp lý.

Ba mốc đo bọc quanh ba module: `dino` (mỗi prompt một lượt), `sam` (`set_image`
và `region_refine`), `saliency`.

### 3.2 Resume theo class, chống lẫn kết quả

Colab ngắt kết nối là mất tiến độ. Nhưng resume ngây thơ lại nguy hiểm hơn: chạy
subset 34 ảnh rồi chạy full, hệ thống thấy "class này có số rồi" và bỏ qua — số
subset bị công bố như số full.

```python
def check_run_identity(meta, meta_key, identity):
    """So sánh danh tính lần chạy hiện tại với entry đã lưu trong run_meta.json.

    So sánh giá trị --max-samples (upper bound của invocation) chứ không phải
    n_images quan sát được: --max-samples là cận trên và stratified subsampling
    có thể cho ra ít ảnh hơn, nên so n_images là mơ hồ còn so flag là chính xác.
    """
    if meta_key not in meta:
        return False, f'không có entry trong run_meta.json cho {meta_key}'

    entry = meta[meta_key]
    for field, expected in identity.items():
        actual = entry.get(field, '<missing>')
        if actual != expected:
            return False, f'{field} lệch: run_meta={actual!r} lần chạy hiện tại={expected!r}'

    return True, ''
```

Danh tính gồm `max_samples`, `cal_pro`, `sam_variant`, `saliency_backbone`,
`detector`, `prompt_source`. Lệch bất kỳ trường nào thì chạy lại chứ không skip.

### 3.3 Bộ test

102 test, chạy trên môi trường không có GPU. Repo gốc có 0 test.

## 4. Phase B — nén pipeline

### 4.1 Thiết kế cho phép hoán đổi

SAM chỉ vào pipeline qua **một điểm khởi tạo** và ba lời gọi (`set_image`,
`transform.apply_boxes_torch`, `predict_torch`). MobileSAM giữ nguyên interface
`SamPredictor`, nên thay là drop-in thật sự.

```python
# SAA/backbones.py — mọi import nặng nằm trong thân hàm, có chủ đích:
# thiếu MobileSAM chỉ làm hỏng nhánh MobileSAM, không hỏng baseline.

def build_sam_predictor(variant, checkpoint, device):
    if variant == 'vit_h':
        from SAM.segment_anything import SamPredictor, build_sam
        return SamPredictor(build_sam(checkpoint=checkpoint).to(device))

    if variant == 'mobile_sam':
        from mobile_sam import SamPredictor, sam_model_registry
        sam = sam_model_registry['vit_t'](checkpoint=checkpoint)
        sam.to(device); sam.eval()
        return SamPredictor(sam)
    ...
```

Mặc định `vit_h` + `wide_resnet50` tái tạo **chính xác** hành vi cũ, nên baseline
giữ bit-exact.

### 4.2 Kết quả profiling

Một class (`carpet`, 117 ảnh):

| Cấu hình | SAM | Saliency | `p_f1` | `t_dino` | `t_sam` | `t_total` | VRAM |
|---|---|---|---|---|---|---|---|
| baseline | ViT-H | WRN-50 | 56.64 | 1837.5 | 2294.8 | 4255.4 | 6498.9 |
| lite1 | MobileSAM | WRN-50 | 55.53 | 1866.2 | 99.2 | 2095.7 | 1817.7 |
| **lite2** | MobileSAM | MobileNetV3 | 55.36 | 1866.3 | 99.0 | **2077.9** | **1734.2** |

```
t_sam    2294.8 → 99.0    23.17×
t_total  4255.4 → 2077.9   2.05×
t_dino   1837.5 → 1866.3   0.98×   ← không đụng tới
```

Trần lý thuyết nếu SAM và saliency về 0: `4255.4 / 1866.3 = 2.28×`. Đạt 2.05×,
tức **đã vắt 90% dư địa**. Không còn gì để lấy thêm từ việc thay SAM.

Dòng `0.98×` ở DINO là bằng chứng đối chứng: phép đo tách giai đoạn không lẫn
thời gian giữa các tầng.

**Saliency chiếm 1% thời gian** — bản thân đó là một kết luận: không phải chỗ đáng
tối ưu.

### 4.3 Trục detector — kết quả âm

Sau khi thấy DINO chiếm 72%, nhóm mở rộng ngoài phạm vi ban đầu để thử thay nó.

**Đây không phải drop-in.** `bbox_suppression` ăn sâu vào ba thứ riêng của DINO:

```python
logits = outputs["pred_logits"].sigmoid()[0]     # (nq, 256) — điểm theo TỪNG TOKEN
tokenlizer = self.anomaly_region_generator.tokenizer
pred_phrase = get_phrases_from_posmap(logit > text_score_thr, tokenized, tokenlizer)
if pred_phrase.count(filtered_phrase) > 0:       # lọc nền
    continue
```

`logits` không phải một điểm tin cậy mỗi box — nó là ma trận `(số query × 256
token BERT)`. Pipeline dùng nó để hỏi ngược: *box này khớp với chữ nào trong câu
prompt?* YOLO-World và OWLv2 chỉ trả về một điểm mỗi (box, query).

Giải pháp: **nhánh song song**, không refactor. Đường DINO giữ nguyên từng dòng;
detector kiểu query đi qua hàm riêng trả về đúng bộ ba mà pipeline đang chờ.

| Cấu hình | Detector | `p_ap` | `p_f1` | `t_total` | VRAM |
|---|---|---|---|---|---|
| lite2 | Grounding DINO | 37.16 | 55.36 | 2077.9 | 1734.2 |
| lite2_owlv2 | OWLv2 | 4.59 | 9.22 | 3981.8 | **923.8** |
| lite2_yolo | YOLO-World | 1.60 | 3.15 | 5636.3 | 1950.4 |

**Cả hai bị chi phối hoàn toàn.** YOLO-World còn chậm hơn baseline ViT-H (0.76×) —
một model 13M tham số thua model 636M về tốc độ.

Hai kết luận **khác bản chất**, không được gộp:

- **Accuracy sụp là do model.** Grounding DINO neo được cụm mô tả defect
  (`"black hole"`, `"thread"`); YOLO-World và OWLv2 huấn luyện để dò **danh từ vật
  thể**, không dò tính từ mô tả khuyết tật.
- **Chậm là do cách tích hợp.** Pipeline gọi detector một lượt mỗi prompt, và cả
  hai đều mã hoá lại văn bản mỗi lượt. Gộp lại sẽ nhanh hơn — nhưng không đáng
  làm, vì kể cả bằng tốc độ DINO thì accuracy vẫn bị chi phối.

Chỗ này phải viết rõ, nếu không hội đồng sẽ hỏi tại sao một detector thời gian
thực lại chậm hơn SAM ViT-H.

### 4.4 Full run và ba đính chính

Cấu hình lite2 trên toàn bộ 27 class:

| | MVTec baseline | lite2 | Giữ | VisA baseline | lite2 | Giữ |
|---|---|---|---|---|---|---|
| `p_ap` | 28.86 | 28.24 | 97.8% | 22.07 | 24.35 | **110.3%** |
| `p_f1` | 37.72 | 37.44 | 99.3% | 33.74 | 34.92 | **103.5%** |
| `t_total` | 3994 | 1926 | **2.07×** | 4756 | 3110 | **1.53×** |

Profiling một class cho ba kết luận, **cả ba đều lệch**:

| | `carpet` (1 class) | Full 15 class |
|---|---|---|
| `p_ap` giữ được | 93.0% — trượt ngưỡng 95% | **97.8%** — đạt |
| DINO chiếm | 90% | **72%** |
| VRAM giảm | 3.75× | 1.86× (trung vị) |

Nguyên nhân của cái thứ hai: `t_sam` biến thiên 98 ms (`carpet`) tới 788 ms
(`pill`) — decoder SAM chạy một lượt mỗi box.

## 5. Phase A — nguồn prompt

### 5.1 Thang so sánh

| Mức | `--prompt-source` | Prompt | Lượt DINO/ảnh |
|---|---|---|---|
| P0 | `generic` | `"defect."` | 2.00 |
| P1 | `general` | 3 `general_prompts` | 4.00 |
| P2b | `llm` | Qwen2.5-VL-3B, chỉ biết tên class | 4.53 |
| P2v | `llm` | Qwen2.5-VL-3B + 2 ảnh normal 512px | 3.40 |
| P3 | `manual` | 3 general + K manual | 5.20 |

### 5.2 Số lượt DINO tỉ lệ thuận với số prompt

| Mức | Lượt/ảnh | `t_dino` | ms/lượt |
|---|---|---|---|
| P0 | 2.00 | 516.7 | 258.4 |
| P1 | 4.00 | 1050.7 | 262.7 |
| P2b | 4.53 | 1204.8 | 265.8 |
| P2v | 3.40 | 871.2 | 256.2 |
| P3 | 5.20 | 1394.1 | 268.1 |

Ổn định qua năm nguồn. `t_dino` chỉ phụ thuộc **số lượt** gọi.

### 5.3 Sinh prompt bằng VLM cục bộ

Không gọi API ngoài. Lý do không chỉ là chi phí:

- **Tái lập được 100%.** Greedy decoding trên trọng số cố định: hội đồng tái sinh
  được chính xác cùng một file JSON. API thì không — model sau một cái tên có thể
  đổi.
- **Tuyên bố mạnh hơn:** "zero-shot không cần API trả phí".

**Ranh giới chống rò rỉ dữ liệu** là phần quan trọng nhất của script:

```python
# tools/gen_prompts.py
def train_image_paths(dataset, class_name, root, limit):
    """Đường dẫn ảnh trong train/good, đã sắp xếp.

    CHỈ train split. Đây là ranh giới chống rò rỉ dữ liệu ở spec mục 5.5.
    Sắp xếp để greedy decoding tái lập được.
    """
    pattern = SPLIT_DIRS[dataset].format(class_name=class_name)
    directory = os.path.join(root, pattern)

    if not os.path.isdir(directory):
        return []

    names = sorted(n for n in os.listdir(directory)
                   if n.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')))

    return [os.path.join(directory, n) for n in names[:limit]]
```

Có test chặn mọi đường tới `test/`, `ground_truth/`, ảnh anomaly.

**Ranh giới thứ hai, dễ sót hơn:** prompt hệ thống **không được chứa phát hiện đo
trên tập test**. Thang so sánh cho thấy texture hợp với prompt generic hơn — viết
điều đó vào bộ sinh prompt là mã hoá kết quả tập test vào chính thứ đang được
đánh giá. Có test grep các từ báo hiệu vượt lằn ranh đó.

### 5.4 Kết quả

| Mức | `p_f1` | so với P3 | Tốc độ |
|---|---|---|---|
| P0 | 35.54 | −1.90 | 2.62× |
| P1 | 35.80 | −1.64 | 1.28× |
| P2b | 31.16 | −6.28 | 1.26× |
| P2v | 29.32 | −8.12 | 1.70× |
| P3 | **37.44** | — | 1.00× |

**Cả hai biến thể LLM thất bại**, mọi ngưỡng đều trượt.

### 5.5 Hai cơ chế hỏng, đặt tên được

**Prompt cụ thể SAI nguy hiểm hơn prompt mơ hồ ĐÚNG.**

`pill`: P0 (`"defect."`) đạt 63.94, P2b (`crack`/`scratch`/`stain`/`missing piece`)
chỉ 11.10 — mất **52.84 điểm**. Defect `pill` trong MVTec là đổi màu và in lỗi.
`"defect."` không mã hoá giả định nào nên không sai được.

**Cho model nhìn ảnh bình thường khiến nó mô tả tính bình thường.**

Dự đoán trước khi chạy: `pill` sẽ tăng ở P2v, vì prompt sinh ra là `red spots` và
`letter F` — hai thứ **có thật** trên ảnh viên thuốc. Thực tế **giảm** (11.10 →
8.08): chúng có thật trên **mọi** viên thuốc, kể cả viên bình thường.

Cùng cơ chế làm `capsule` sập 21.89 điểm: prompt là `['500', 'capsule']`, trong đó
`500` là liều lượng in trên vỏ nang, còn `capsule` là chính vật thể nên bị bộ lọc
nền loại. Class đó còn **0 prompt dùng được**.

> Cho model nhìn ảnh **bình thường** thì nó mô tả **tính bình thường**. Anomaly
> detection cần **cái không nên có**.

### 5.6 Kết quả trung tâm: không nguồn nào thống trị

```
Nguồn thắng ở từng class:  P1 4   P0 3   P2b 3   P3 3   P2v 2
```

| Chọn giữa | `p_f1` | Vượt P3 |
|---|---|---|
| P3 một mình (chuyên gia) | 37.44 | — |
| oracle-3 (P0/P1/P3) | 39.77 | +2.33 |
| oracle-4 (+P2b) | 40.80 | +3.36 |
| **oracle-5 (+P2v)** | **41.06** | **+3.62** |

Biến thiên theo class rất lớn: `pill` có P0 hơn P3 **19 điểm**, `cable` có P3 hơn
P0 **18 điểm**.

Bảng đầy đủ theo class:

| Class | P0 | P1 | P2b | P2v | P3 | Loại | Thắng |
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

### 5.7 Texture và object đi ngược chiều

```
texture   P3 − P1 = −1.07 điểm     prompt thủ công LÀM HẠI
object    P3 − P1 = +3.00 điểm     prompt thủ công giúp
```

Cơ chế: defect trên bề mặt vân là "chỗ nào khác phần còn lại", mà
`"defect on carpet"` diễn tả đúng thế. Defect trên vật thể có tên cụ thể mà prompt
generic không gọi ra được.

## 6. Kiểm chứng metric max-F1-region

Paper định nghĩa (mục 5.1):

> *we compute the F1-score for region-wise segmentation at the optimal threshold,
> considering a prediction positive if the overlapping value exceeds 0.6*

F1 theo định nghĩa này **không thể vượt 1**: TP là một con số duy nhất, dùng chung
cho cả precision và recall, nên `TP ≤ min(n_pred, n_gt)`.

Cài đặt trong repo trả về **122.57** cho class `wood`. Tái hiện:

```python
gt = np.zeros((1, 60, 60), dtype=np.uint8)
gt[0, 20:40, 20:40] = 1                    # một vùng GT liền khối

scores = np.zeros((1, 60, 60))
scores[0, 20:40, 20:40] = 1.0
scores[0, 29:31, 20:40] = 0.0              # khe 1 pixel tách dự đoán làm đôi

calculate_max_f1_region(gt, scores)        # → 1.3333
```

Trên fixture nhiễu Gauss, hàm trả về `recall = 2416.5`.

**Hai lỗi lồng nhau:**

```python
# A — cửa sổ tính overlap bị nhiễm
cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
# cắt TOÀN BỘ bản đồ nhị phân theo bbox hợp, không phải riêng vùng đang xét

# B — recall chia sai mẫu số
hits = (pro >= pro_thresh).sum()    # đếm vùng DỰ ĐOÁN khớp được
recall = hits / gt_region_number    # chia cho số vùng GT
```

B là hệ quả của A: cửa sổ nhiễm khiến cả hai mảnh dự đoán đều đạt IoU > 0.6.

Chính tác giả để lại dấu vết ngay chỗ đó: `# cropped_mask = prop.filled_image  # corrected!`

**Bản đúng** (`calculate_max_f1_region_fixed`) lấy TP từ phép ghép một-một và tính
IoU trên mặt nạ từng vùng:

```python
def _count_matched_pairs(pairs, pro_thresh):
    """Ghép một-một tham lam theo IoU giảm dần, trả về số cặp khớp được.

    Một-một là chỗ quyết định: nó đảm bảo TP <= min(số vùng dự đoán, số vùng GT),
    nên precision và recall đều <= 1 và F1 không thể vượt 1.
    """
    matched_pred, matched_gt = set(), set()
    true_positives = 0

    for iou, p, g in sorted(pairs, reverse=True):
        if iou < pro_thresh:
            break
        if p in matched_pred or g in matched_gt:
            continue
        matched_pred.add(p); matched_gt.add(g)
        true_positives += 1

    return true_positives
```

Bản cũ thổi phồng ~2× trên MVTec (42.45 so với 21.92). Bản đúng còn **nhanh gấp
25 lần** vì bỏ được vòng cắt bbox chồng chéo.

**Xử lý:** báo cáo cả hai cột. `r_f1` để so với bảng đã công bố, `r_f1_fixed` là
số đúng.

## 7. Bốn lỗi tiềm ẩn phát hiện trong repo gốc

Ba lỗi cuối chỉ lộ ra khi đi qua nhánh code mà đường baseline không bao giờ chạm.

| Lỗi | Vị trí | Triệu chứng |
|---|---|---|
| `calculate_max_f1_region` recall > 1 | `utils/metrics.py` | `r_f1` = 122.57 cho `wood` |
| Nhánh không còn box trả `list` thay vì `ndarray` | `SAA/model.py` | `TypeError: list indices must be integers` |
| `normalize()` chia cho 0 khi map hằng số | `utils/eval_utils.py` | `ValueError: Input contains NaN` sau 11 phút inference |
| `--vis` khai nhưng `is_vis=True` cố định | `eval_SAA.py` | cờ chưa bao giờ có tác dụng |

Hai lỗi giữa chỉ lộ ra khi thử detector khác — giá trị phụ của kết quả âm.

## 8. Điều tra chênh lệch VisA

Baseline trên VisA cao hơn paper **24.6%**. Nhóm loại được bốn nguyên nhân:

- **Không phải split.** Loader mặc định `VisA_pytorch/1cls`; số ảnh test khớp kỳ
  vọng, 11/12 class tuyệt đối.
- **Không phải prompt bị sửa.** `visa_parameters.py` có đúng một commit, lần import
  gốc.
- **Không phải lệch hệ thống.** Chênh lệch ở 2/12 class: `chewinggum` 86.12 và
  `capsules` 59.27. Bỏ hai class đó, mean còn 25.95 so với 27.07 của paper.
- **Không phải resize, không riêng một metric.** `r_f1` lệch cùng hướng.

**Còn lại:** prompt trong repo có thể khác prompt đã tạo ra Table 1. Paper đẩy chi
tiết prompt sang supplementary material, không có trong repo.

**Nhóm không tuyên bố tái lập thành công trên VisA**, và cũng không lảng tránh con
số.

## 9. Hạn chế

1. **VisA chưa tái lập được** — xem mục 8.
2. **lite2 tốt hơn baseline trên VisA** (110.3% `p_ap`) — chưa giải thích được.
   Giả thuyết: mask thô hơn của MobileSAM khớp hơn với defect lớn, mờ ranh giới.
3. **Chưa kiểm bit-exact bằng thực nghiệm.** Việc factory backbone không đổi hành
   vi hiện mới được xác minh bằng đọc code và phép kiểm AST.
4. **Nhiễm dữ liệu huấn luyện.** SAA+ là repo công khai, paper đăng IEEE; VLM có
   thể đã thấy prompt của tác giả. Đối chiếu từng chữ cho **0/53** (blind) và
   **0/36** (vision) — không có dấu hiệu nhớ trực tiếp, nhưng không loại trừ được
   ảnh hưởng gián tiếp.
5. **Phase A chỉ chạy trên MVTec**, chưa chạy VisA.
6. **Chỉ dùng một VLM** (Qwen2.5-VL-3B). Model lớn hơn có thể cho kết quả khác.
7. **Số latency chỉ so sánh được trong cùng một loại GPU.** Toàn bộ đo trên T4.

## 10. Hướng phát triển

- Trả lời ba câu hỏi còn mở ở mục 9
- Phase A trên VisA
- Thử VLM lớn hơn cho phần sinh prompt
- **Bộ chọn nguồn prompt theo từng class** — dữ liệu cho thấy dư địa 3.62 điểm,
  và đây là hướng có cơ sở thực nghiệm rõ nhất
- Web demo FastAPI trên cấu hình lite2

---

# PHẦN III — SLIDES

> Ghép file PDF slide đã báo cáo giữa kỳ vào đây. Dưới là những chỗ cần cập nhật
> theo kết quả mới.

## Slide cần thêm hoặc sửa

| Slide | Nội dung |
|---|---|
| Kết quả Phase B | Bảng lite2: giữ 99.3% `p_f1`, nhanh 2.07× |
| **Biểu đồ Pareto backbone** | `p_f1` theo `t_total`, 5 điểm, kích thước điểm theo VRAM |
| DINO là nút cổ chai | Biểu đồ cột tỉ trọng: baseline 34% → lite2 72% |
| Trục detector — kết quả âm | Bảng 3 dòng, nhấn YOLO-World chậm hơn cả ViT-H |
| **Biểu đồ Pareto nguồn prompt** | 5 điểm P0-P3 + đường oracle 41.06 |
| Không nguồn nào thống trị | Bar chart per-class, tô màu theo nguồn thắng |
| Hai cơ chế hỏng | `pill` −52.84 điểm; `capsule` → `500` |
| Lỗi max-F1-region | Ví dụ tái hiện 5 dòng + con số 122.57 |
| Hạn chế | Mục 9 phần II |

## Ba biểu đồ cần vẽ

1. **Pareto backbone** — trục X `t_total` (ms), trục Y `p_f1`, 5 điểm
   (baseline, lite1, lite2, yolo, owlv2), kích thước điểm tỉ lệ `peak_vram`
2. **Pareto nguồn prompt** — 5 điểm P0/P1/P2b/P2v/P3 + đường ngang oracle 41.06
3. **Per-class** — bar chart 15 class, mỗi class 5 cột, tô đậm cột thắng

---

## Phụ lục — Dữ liệu và tái lập

| | |
|---|---|
| Báo cáo tổng hợp | `docs/report.md` |
| Baseline full-run | `results/baseline_full/` |
| Profiling + trục detector | `results/profiling_step1/` |
| Full-run lite2 | `results/lite2_full/` |
| Thang nguồn prompt | `results/phase_a_ladder/` |
| Prompt do LLM sinh | `SAA/prompts/generated/` |
| Thiết kế thí nghiệm | `docs/superpowers/specs/` |

Notebook: `Benchmark_SAA.ipynb` (baseline), `Profiling_SAA_Lite.ipynb` (Bước 1),
`Benchmark_SAA_Lite2.ipynb` (Bước 2), `Phase_A_Prompt_Ladder.ipynb` (thang prompt),
`Generate_Prompts_VLM.ipynb` và `Generate_Prompts_VLM_Vision.ipynb` (sinh prompt).

Mỗi lần chạy benchmark đều có `run_meta.json` ghi loại GPU, số ảnh và cấu hình.
Prompt do LLM sinh có `*-meta.json` ghi model, quantization, số ảnh, kích thước
ảnh, và nguyên văn prompt đã dùng.
