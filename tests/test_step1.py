"""1단계 정합성 테스트 — 뼈대. 구현하면서 하나씩 통과시킨다.

이 테스트들이 통과하기 전의 성능 숫자는 믿지 않는다.
"""
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skip(reason="TODO: src/ 구현하면서 하나씩 skip 을 푼다")


def test_ridge_alpha_zero_equals_ols():
    """λ=0 ridge == OLS. 표준화·절편 처리가 맞는지 보는 가장 강한 단일 테스트.
    TODO: 난수 X, y 를 만들고 statsmodels OLS 와 계수·적합값을 atol=1e-10 로 비교한다."""


def test_ridge_shrinks_with_alpha():
    """λ 가 커지면 ‖coef‖ 가 단조 감소한다. TODO"""


def test_lasso_zeroes_out_noise_columns():
    """신호 2열 + 잡음 8열을 만들면 lasso 가 잡음 열 계수를 0 으로 보낸다. TODO"""


def test_scaler_uses_train_only():
    """TrainScaler 를 앞 절반으로 fit 한 뒤 뒤 절반을 transform 했을 때,
    변환된 뒤 절반의 평균이 0 이 **아니어야** 한다. 0 이면 전체 표본으로 스케일링한 것 = 누수. TODO"""


def test_expanding_splits_never_see_future():
    """모든 분할에서 max(train) < predict_at. TODO"""


def test_walk_forward_matches_source_ols():
    """배관 검증용 회귀 테스트 — **여기서만** include_ar=True 를 쓴다.

    원본 factor-nowcasting 의 `ols` 벤치마크는 y_q = c + φ y_{q−1} + b'F_q 다.
    같은 설정으로 walk_forward 를 돌려 backtest_preds_<code>.csv 의 `ols` 열을 1e-8 이내로
    재현하면, 내 파이프라인이 원본과 같은 정보집합(훈련 구간·시차·dropna)을 쓴다는 증거가 된다.
    연구용 설정(AR 항 없음)과는 별개다. 원천 저장소가 없으면 skip. TODO"""
