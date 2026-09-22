"""베이스라인 — 뼈대. **ML 보다 먼저 만든다.**

베이스라인 없는 R² 는 숫자가 아니다. 팩터 nowcasting 에서 실제로 벌어진 일:
Burgiss 부동산은 AR(1) 이 4팩터 회귀를 이긴다. 감정평가 지수라 자기상관이 신호의 대부분이기
때문이다. 베이스라인을 먼저 두지 않았으면 "R² 0.14, 팩터가 먹는다" 로 잘못 읽었을 것이다.

세 가지를 둔다.
  mean : 확장창 평균. 가장 약한 기준. R²_OOS 의 분모가 된다 (Campbell-Thompson).
  ar1  : y_q = c + φ y_{q−1}. **진단용이지 후보 모델이 아니다.**
         자기 과거만 쓰므로 공모자산과 같은 자에 놓을 수 없다. 그래도 필요한 이유는
         이 값이 "이 시리즈가 감정평가 스무딩에 얼마나 지배되는가" 의 눈금이기 때문이다.
         ar1 의 R²_OOS 가 크면 그 시리즈는 팩터가 아니라 관성이 움직인다는 뜻이다.
  ols  : y_q = c + b'F_q. **동분기 팩터만, AR 항 없음.** 정규화 없는 원형(原型) 이자
         공모자산 모델과 같은 형태다. Ridge/Lasso 는 이 식에 벌점을 붙인 것이므로
         이것이 진짜 비교 대상이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def predict_mean(y_train: pd.Series) -> float:
    """확장창 평균. TODO: 한 줄."""
    raise NotImplementedError


def predict_ar1(y_train: pd.Series) -> float:
    """AR(1) 진단 베이스라인. y_train 으로 적합하고 다음 한 분기를 예측한다.

    y_t = c + φ y_{t−1} 을 OLS 로 적합 → 예측값 = c + φ × y_train 의 마지막 값.
    TODO: np.linalg.lstsq 로 직접 풀어 본다 (statsmodels 를 쓰기 전에 손으로 한 번).
    """
    raise NotImplementedError


def predict_ols(y_train: pd.Series, fq: pd.DataFrame, predict_at: pd.Timestamp) -> float:
    """y_q = c + b'F_q. 동분기 팩터만 쓴다 (AR 항 없음). 훈련은 y_train 구간, 예측은 predict_at 분기.

    이 식의 계수 b 가 곧 **보고 베타**다. 공모자산에 같은 회귀를 돌려 나온 베타와 나란히 놓을 수 있다.
    다만 감정평가 스무딩이 있으면 b 가 실제 노출보다 작게 나온다 (README 의 '베타 감쇠' 참고).

    TODO: y_train 과 fq 를 concat → dropna → np.linalg.lstsq → np.r_[1.0, fq.loc[predict_at]] @ coef.
    """
    raise NotImplementedError
