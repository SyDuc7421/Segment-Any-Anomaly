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


class RecordingClock:
    """Logs clock reads and sync calls into one shared, ordered list."""

    def __init__(self, ticks):
        self.ticks = list(ticks)
        self.calls = 0
        self.events = []

    def clock(self):
        self.events.append('clock')
        value = self.ticks[self.calls]
        self.calls += 1
        return value

    def sync(self):
        self.events.append('sync')


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


def test_summarize_timings_averages_across_images_excluding_warmup():
    """Anh dau tien bi loai khoi trung binh (warm-up: lazy CUDA init, cuDNN
    autotune, first-touch allocation). n_images van bao cao dung so anh that
    su da chay, kho voi so anh dung de tinh trung binh thoi gian."""
    records = [
        {'dino': 100.0, 'sam': 40.0, 'saliency': 10.0, 'total': 150.0},
        {'dino': 200.0, 'sam': 60.0, 'saliency': 30.0, 'total': 290.0},
    ]

    summary = summarize_timings(records, n_images=2, peak_vram_mb=1234.5)

    assert summary == {
        't_dino': 200.0,
        't_sam': 60.0,
        't_saliency': 30.0,
        't_total': 290.0,
        'n_images': 2,
        'peak_vram': 1234.5,
    }


def test_summarize_timings_drops_first_record_as_warmup():
    """Ghim ro rang hanh vi drop-first: ban ghi dau tien co gia tri rat lon
    (mo phong warm-up chi phi cao) khong duoc phep keo lech trung binh."""
    records = [
        {'dino': 5000.0, 'sam': 5000.0, 'saliency': 5000.0, 'total': 15000.0},
        {'dino': 100.0, 'sam': 40.0, 'saliency': 10.0, 'total': 150.0},
        {'dino': 200.0, 'sam': 60.0, 'saliency': 30.0, 'total': 290.0},
    ]

    summary = summarize_timings(records, n_images=3, peak_vram_mb=0.0)

    assert summary['t_dino'] == pytest.approx(150.0)
    assert summary['t_sam'] == pytest.approx(50.0)
    assert summary['t_saliency'] == pytest.approx(20.0)
    assert summary['t_total'] == pytest.approx(220.0)
    assert summary['n_images'] == 3


def test_summarize_timings_uses_the_single_record_when_only_one_exists():
    """Chi co 1 anh thi khong the loai bo warm-up (se ra toan so 0), nen dung
    luon ban ghi do — tot hon la bao cao 0."""
    records = [{'dino': 100.0, 'sam': 40.0, 'saliency': 10.0, 'total': 150.0}]

    summary = summarize_timings(records, n_images=1, peak_vram_mb=0.0)

    assert summary['t_dino'] == pytest.approx(100.0)
    assert summary['t_sam'] == pytest.approx(40.0)
    assert summary['t_saliency'] == pytest.approx(10.0)
    assert summary['t_total'] == pytest.approx(150.0)
    assert summary['n_images'] == 1


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


def test_nested_stages_record_separately_while_outer_spans_the_whole_block():
    """Cach StageTimer thuc su duoc dung trong SAA/model.py: dino, sam, va
    saliency deu chay ben trong total. Moi stage phai cong don rieng, con
    total phai bao trum toan bo khoi ke ca khoang cach giua cac stage con."""
    timer = StageTimer(clock=FakeClock([0.0, 0.1, 0.4, 0.5, 0.9, 1.0]))

    with timer.stage('total'):
        with timer.stage('dino'):
            pass
        with timer.stage('sam'):
            pass

    assert timer.snapshot() == pytest.approx({'total': 1000.0, 'dino': 300.0, 'sam': 400.0})


def test_sync_precedes_every_clock_read():
    recorder = RecordingClock([0.0, 1.0])

    timer = StageTimer(clock=recorder.clock, sync=recorder.sync)
    with timer.stage('sam'):
        pass

    assert recorder.events == ['sync', 'clock', 'sync', 'clock']
    assert timer.snapshot() == {'sam': 1000.0}
