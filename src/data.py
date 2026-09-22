"""데이터 적재 — scripts/build_dataset.py 가 만든 분기 패널을 읽는다. (배관이라 완성 코드)

패널 규약
  · 인덱스: 분기말 (2000Q1~2026Q2). 대체투자 이력이 팩터보다 길어 앞 구간은 팩터가 NaN 이다.
  · 팩터 4개(growth/term/inflation/credit): 일간 단순수익률의 분기 복리 Π(1+r)−1. 부분 분기(2006Q1,
    진행 중 분기)는 제외돼 있다.
  · 대체투자 15 시리즈: 보고된 분기 단순수익률. 시리즈마다 시작·종료 분기가 다르다.
  · 부호: term + = 금리 하락, inflation + = 기대인플레 상승, credit + = 스프레드 축소.
"""
from __future__ import annotations

import json

import pandas as pd

from .config import load_paths, resolve

FACTORS = ["growth", "term", "inflation", "credit"]


def load_panel() -> pd.DataFrame:
    p = resolve(load_paths()["processed"]) / "panel_quarterly.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} 없음 — 먼저 `python scripts/build_dataset.py` 를 돌린다")
    return pd.read_csv(p, index_col=0, parse_dates=True).sort_index()


def load_meta() -> pd.DataFrame:
    m = pd.read_csv(resolve(load_paths()["processed"]) / "series_meta.csv", index_col=0)
    m["labels"] = m["labels"].str.split("|")
    return m


def panel_meta() -> dict:
    return json.loads((resolve(load_paths()["processed"]) / "panel_meta.json").read_text(encoding="utf-8"))


def get_target(panel: pd.DataFrame, code: str) -> pd.Series:
    """시리즈 code 의 보고 분기수익률. 앞뒤 결측을 떼고 반환한다."""
    if code not in panel.columns:
        raise KeyError(f"{code} 없음. 가능: {[c for c in panel.columns if c not in FACTORS]}")
    return panel[code].dropna()


def get_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """팩터가 모두 있는 분기만 (= 완전 분기)."""
    return panel[FACTORS].dropna(how="any")
