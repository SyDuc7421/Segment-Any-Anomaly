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

    Bỏ ảnh đầu tiên khỏi trung bình khi có nhiều hơn 1 bản ghi: ảnh đầu
    "gánh" lazy CUDA init, cuDNN autotune, và first-touch allocation, nên
    thời gian đo được lệch cao so với các ảnh sau — trên vài chục ảnh của
    một lần profiling ngắn (spec mục 6.2 Bước 1), lệch này có thể tới hàng
    chục phần trăm. Hệ quả: t_dino/t_sam/t_saliency/t_total là trung bình
    trên (N-1) ảnh, trong khi n_images vẫn báo cáo đúng N ảnh thực sự đã
    chạy — hai con số này KHÔNG cùng mẫu số.

    Args:
        records: danh sách dict trả về từ StageTimer.snapshot(), mỗi ảnh một dict.
        n_images: số ảnh thực sự đã chạy trong lần chạy này.
        peak_vram_mb: đỉnh VRAM, đơn vị MB.
    """
    summary = {}

    if len(records) > 1:
        timed_records = records[1:]
    else:
        timed_records = records

    divisor = len(timed_records) if timed_records else 1

    for stage_name, column in STAGE_TO_COLUMN:
        total = sum(record.get(stage_name, 0.0) for record in timed_records)
        summary[column] = total / divisor if timed_records else 0.0

    summary['n_images'] = n_images
    summary['peak_vram'] = peak_vram_mb

    return summary
