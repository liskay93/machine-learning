"""베이스라인 — 뼈대. **ML 보다 먼저 만든다.**

베이스라인 없는 R² 는 숫자가 아니다. 팩터 nowcasting 에서 실제로 벌어진 일:
Burgiss 부동산은 AR(1) 이 4팩터 회귀를 이긴다. 감정평가 지수라 자기상관이 신호의 대부분이기
때문이다. 베이스라인을 먼저 두지 않았으면 "R² 0.14, 팩터가 먹는다" 로 잘못 읽었을 것이다.

세 가지를 둔다 — 점점 강해지는 순서다.
  mean : 확장창 평균. 가장 약한 기준. R²_OOS 의 분모가 된다 (Campbell-Thompson).
  ar1  : y_q = c + φ y_{q−1}. 스무딩된 지수에서 강하다.
  ols  : y_q = c + φ y_{q−1} + b'F_q. 동분기 팩터 + AR. 정규화 없는 원형(原型).
         Ridge/Lasso 는 결국 이 식에 벌점을 붙인 것이므로, 이것이 진짜 비교 대상이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def predict_mean(y_train: pd.Series) -> float:
    """확장창 평균. TODO: 한 줄."""
    raise NotImplementedError


def predict_ar1(y_train: pd.Series) -> float:
    """AR(1) 을 y_train 으로 적합하고 다음 한 분기를 예측한다.

    y_t = c + φ y_{t−1} 을 OLS 로 적합 → 예측값 = c + φ × y_train 의 마지막 값.
    TODO: np.linalg.lstsq 로 직접 풀어 본다 (statsmodels 를 쓰기 전에 손으로 한 번).
    """
    raise NotImplementedError


def predict_ols(y_train: pd.Series, fq: pd.DataFrame, predict_at: pd.Timestamp) -> float:
    """y_q = c + φ y_{q−1} + b'F_q. 훈련은 y_train 구간, 예측은 predict_at 분기.

    주의: 설계행렬을 만들 때 y_train.shift(1) 은 y 자신의 인덱스 위에서 한다 (features.build_design 의 주의와 같다).
    TODO: concat → dropna → lstsq → x_pred @ coef.
    """
    raise NotImplementedError
