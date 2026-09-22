"""선형 모델 — OLS → Ridge → Lasso → Elastic Net. 1단계의 본체.

아는 것에 붙여서 이해하기
  · **Ridge = 베이지안 prior.** 계수에 β ~ N(0, τ²) 을 얹고 MAP 를 구하면 정확히 ridge 가 된다.
    λ = σ²/τ². λ 가 크다 = "계수가 0 근처일 것이다" 라는 믿음이 세다 = 자료를 덜 믿는다.
    표본 57분기에 변수 21개인 상황에서, prior 없이 OLS 를 돌리면 자료가 잡음까지 설명해 버린다.
  · **Lasso = L1.** 계수를 0 으로 **딱** 보낸다 (ridge 는 0 에 가깝게만). 변수 선택이 공짜로 따라온다.
    팩터 lag 1~4 가 쓸모없다면 lasso 가 그것을 0 으로 죽여서 보여 줄 것이다 — 가설 검정을 대신한다.
  · **Elastic Net = 둘의 혼합.** α(1−ρ)/2·‖β‖² + αρ·‖β‖₁. 설명변수끼리 상관이 높을 때
    (growth 와 credit 은 위기 때 같이 움직인다) lasso 는 그중 하나만 임의로 고르고 나머지를 죽인다.
    ridge 성분이 섞이면 상관된 변수들을 함께 살려 준다.
  · **벌점은 절편에 걸지 않는다.** 절편은 단위(평균 수익률)라 줄일 이유가 없다.
  · **표준화 후 벌점을 건다.** 안 하면 단위가 큰 변수만 벌을 받는다.

만드는 순서 (반드시 지킨다)
  1. OLS 를 먼저 돌려 베이스라인 대비 성능을 본다.
  2. λ = 0 일 때 ridge 가 OLS 와 **수치적으로 같은지** 테스트로 확인한다 (tests/test_models_linear.py).
     이게 통과하지 않으면 그 뒤 비교는 전부 의미가 없다.
  3. λ 를 훈련창 **안쪽** CV 로 고른다 (src/cv.inner_splits).
  4. 선택된 λ 가 시간에 따라 어떻게 움직이는지 그려 본다 — 표본이 늘면 λ 는 보통 줄어든다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PenalizedLinear:
    """OLS/Ridge/Lasso/ElasticNet 하나로 다루는 래퍼. 안쪽 시간순 CV 로 하이퍼파라미터를 고른다.

    kind : "ols" | "ridge" | "lasso" | "enet"
    alphas / l1_ratios : 후보 격자. kind="ols" 면 무시한다.
    inner_min_train : 안쪽 CV 의 첫 검증점 이전 최소 관측 수.

    적합 후 채워지는 것: alpha_, l1_ratio_, coef_(pd.Series), intercept_, cv_score_(격자별 SSE), resid_sd_

    설계 원칙
      · 표준화는 이 클래스 **안에서** 훈련 표본으로만 한다 (features.TrainScaler 재사용).
        바깥에서 미리 표준화해 넘기면 walk-forward 각 스텝의 훈련 표본이 달라 누수가 생긴다.
      · sklearn 의 Ridge/Lasso/ElasticNet 을 써도 좋다. 다만 ridge 는 닫힌 해가 있으니
        (X'X + λI)⁻¹X'y 를 한 번은 직접 풀어 보고 sklearn 결과와 맞춰 본다.
      · sklearn 의 RidgeCV/LassoCV 의 기본 cv 는 KFold 다 — 그대로 쓰면 이 저장소 규칙 위반이다.
        cv 인자에 시계열 분할을 명시하거나 직접 루프를 돈다.
    """

    kind: str = "ridge"
    alphas: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0])
    l1_ratios: list[float] = field(default_factory=lambda: [0.5])
    inner_min_train: int = 20
    standardize: bool = True

    alpha_: float | None = None
    l1_ratio_: float | None = None
    coef_: pd.Series | None = None
    intercept_: float | None = None
    cv_score_: pd.DataFrame | None = None
    resid_sd_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PenalizedLinear":
        """TODO
        1) 표준화 (훈련 표본 기준)
        2) kind 가 ols 면 격자 없이 바로 적합
        3) 아니면 (alpha, l1_ratio) 격자 × inner_splits 로 SSE 를 모아 최소인 조합 선택
        4) 그 조합으로 전체 훈련 표본 재적합 → coef_, intercept_, resid_sd_
        """
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """TODO: 저장된 스케일러로 transform 후 X@coef + intercept."""
        raise NotImplementedError

    def coef_unscaled(self) -> pd.Series:
        """원 단위 계수 (수익률/수익률). 표준화했다면 coef_ / sd_.
        해석할 때는 이 값을 쓴다 — 표준화 계수는 '1 표준편차 충격당' 이라 단위가 다르다.
        TODO
        """
        raise NotImplementedError


def walk_forward(X: pd.DataFrame, y: pd.Series, model_factory, oos_start, min_train: int,
                 embargo: int = 0) -> pd.DataFrame:
    """확장창 walk-forward 로 한 모델의 OOS 예측을 모은다.

    model_factory : 인자 없이 호출하면 **새** 모델 인스턴스를 주는 함수 (매 스텝 새로 적합해야 한다.
                    같은 객체를 재사용하면 앞 스텝의 스케일러·계수가 남아 조용히 누수된다).
    반환: 인덱스 = 예측 분기, 열 = [y, pred, alpha, l1_ratio, n_train]

    TODO: cv.expanding_splits 로 돌면서 매 스텝 leakage_checklist 를 통과시키고 예측을 모은다.
    """
    raise NotImplementedError
