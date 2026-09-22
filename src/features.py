"""설계행렬 만들기 — 뼈대. 채우는 것은 본인 몫이다.

왜 여기서 막히면 안 되나: 누수의 90%는 모델이 아니라 이 파일에서 들어온다.
  · lag 0 팩터(F_q)를 쓰는 것은 **nowcast 라서** 허용된다. 분기말에 팩터는 이미 관측된다.
    같은 행렬로 y_{q+1} 을 맞히려 들면 그 순간 look-ahead 가 된다.
  · 표준화 평균·표준편차는 **훈련 표본에서만** 계산한다. 전체 표본으로 스케일링하면
    테스트 구간의 분산 정보가 훈련으로 새어 들어온다. sklearn 의 StandardScaler 를
    fit 한 뒤 transform 하는 것과 같은 규율이다.
  · dropna 순서가 중요하다. lag 를 만든 뒤 dropna 해야 앞 L 분기가 정확히 빠진다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def lag_panel(fq: pd.DataFrame, lags: int) -> pd.DataFrame:
    """분기 팩터 → lag 0..L 패널.

    반환: 인덱스는 fq 와 같고 컬럼은 `<factor>_l<k>` (예: growth_l0, growth_l1, ...).
    lag k 열의 t 행 값은 fq 의 t−k 행 값이다.

    TODO: shift(k) 로 만든다. 컬럼 순서는 factor 우선/ lag 우선 중 아무거나 좋으나
          build_design 과 일관되게 유지한다.
    """
    raise NotImplementedError


def build_design(fq: pd.DataFrame, y: pd.Series, lags: int, include_ar: bool) -> tuple[pd.DataFrame, pd.Series]:
    """(X, y) 를 만든다. X 와 y 가 모두 관측된 분기만 남긴다.

    X 열 = lag_panel(fq, lags) 의 모든 열 + (include_ar 이면) `y_l1` = y.shift(1).

    주의: y 는 시리즈마다 시작 분기가 다르고 팩터보다 이력이 길다. y.shift(1) 은
    **팩터 패널이 아니라 y 자신의 분기 인덱스** 위에서 해야 2006Q2 의 y_{q−1}(=2006Q1)이 살아남는다.
    이 한 행 때문에 factor-nowcasting 의 OLS 벤치마크와 결과가 달라진다 — 실제로 확인한 함정이다.

    TODO: lag_panel + AR 항을 concat 하고 dropna 한 뒤 X, y 로 쪼갠다.
    """
    raise NotImplementedError


class TrainScaler:
    """훈련 표본 기준 표준화. sklearn StandardScaler 의 축소판을 직접 만들어 본다.

    왜 직접 만드나: fit 이 '훈련 구간에서만' 일어난다는 것을 손으로 확인하기 위해서다.
    벌점(Ridge/Lasso)은 계수 크기에 벌을 주므로, 변수 스케일이 다르면 벌점이 불공평해진다.
    표준화하면 모든 변수가 같은 자에 놓인다.

    TODO: fit(X) 에서 self.mu_, self.sd_ 를 저장하고 (sd_ == 0 은 1 로), transform 에서 적용한다.
          fit_transform 도 있으면 편하다.
    """

    mu_: np.ndarray | None = None
    sd_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> "TrainScaler":
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
