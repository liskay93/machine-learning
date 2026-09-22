"""평가 지표 — 뼈대. 모델을 만들기 **전에** 무엇으로 이길지 정한다.

자산배분 맥락에서 무엇을 볼지:
  rmse      : 큰 오차에 벌을 준다. 분기수익률 단위(%) 로 읽으면 직관적이다.
  mae       : 이상치에 둔하다. rmse 와 크게 갈리면 소수 분기가 성과를 지배한다는 뜻.
  hit       : 부호 적중률. 방향만 맞으면 되는 의사결정(오버/언더웨이트)에 맞는다.
  r2_oos    : 1 − SSE(모델)/SSE(벤치마크). Campbell-Thompson. **0 보다 커야 의미가 있다.**
              벤치마크가 무엇인지 반드시 같이 적는다. 대(對) 평균과 대(對) AR(1) 은 전혀 다른 이야기다.
  dm        : Diebold-Mariano. RMSE 차이가 표본 운인지 본다. OOS 34~57분기면 검정력이 낮다 — 참고용.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """TODO"""
    raise NotImplementedError


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    """TODO"""
    raise NotImplementedError


def hit_ratio(y: np.ndarray, yhat: np.ndarray) -> float:
    """부호 적중률. TODO: y 가 0 인 경우를 어떻게 셀지 정하고 docstring 에 적는다."""
    raise NotImplementedError


def r2_oos(y: np.ndarray, yhat: np.ndarray, y_bench: np.ndarray) -> float:
    """1 − SSE(모델)/SSE(벤치마크). TODO"""
    raise NotImplementedError


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """d_t = e1²−e2² 의 평균이 0 인지 검정. HAC 분산(lag h−1), 양측 p.
    음수면 모델 1 이 낫다. TODO: 분산이 0 이하일 때 NaN 을 돌려준다.
    """
    raise NotImplementedError


def summarize(y: pd.Series, preds: pd.DataFrame, bench: str = "mean", dm_vs: str = "ols") -> pd.DataFrame:
    """모델 × 지표 표. preds 는 (예측 분기 × 모델) 프레임.
    TODO: 위 함수들을 엮는다. r2_oos 는 bench 열 기준, dm 은 dm_vs 열 기준.
    """
    raise NotImplementedError
