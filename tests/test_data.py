"""데이터 계층 정합성 — 이미 구현된 부분이라 지금 통과해야 한다."""
import pandas as pd
import pytest

from src.data import FACTORS, get_factors, get_target, load_meta, load_panel, panel_meta


@pytest.fixture(scope="module")
def panel():
    return load_panel()


def test_panel_is_quarter_end_and_sorted(panel):
    assert panel.index.is_monotonic_increasing
    assert not panel.index.has_duplicates
    assert all(d == d + pd.offsets.QuarterEnd(0) for d in panel.index)


def test_factors_have_no_gaps(panel):
    """팩터 구간은 분기가 하나도 빠지지 않아야 한다 (부분 분기는 애초에 제외됨)."""
    f = get_factors(panel)
    assert list(f.index) == list(pd.date_range(f.index[0], f.index[-1], freq="QE"))
    assert set(f.columns) == set(FACTORS)


def test_partial_quarters_excluded(panel):
    """2006Q1(팩터 시작이 2/1)과 진행 중 분기는 팩터 패널에 없어야 한다."""
    f = get_factors(panel)
    assert pd.Timestamp("2006-03-31") not in f.index
    meta = panel_meta()
    assert f.index[-1] == pd.Timestamp(meta["last_factor_quarter"])


def test_every_series_maps_to_labels(panel):
    meta = load_meta()
    assert set(meta.index) == {c for c in panel.columns if c not in FACTORS}
    assert sum(len(v) for v in meta["labels"]) == 21


def test_target_is_nonempty_and_within_panel(panel):
    for code in load_meta().index:
        y = get_target(panel, code)
        assert len(y) > 30
        assert y.notna().all()
