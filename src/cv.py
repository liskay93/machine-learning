"""시계열 교차검증 — 뼈대.

무작위 KFold 는 이 저장소에서 금지다. 이유는 셋이다.
  1. 미래로 과거를 맞히게 된다 (look-ahead).
  2. 분기 수익률은 자기상관이 있어 이웃 분기가 거의 같은 정보다. 무작위로 쪼개면
     훈련과 검증이 사실상 같은 표본이 되어 성능이 부풀려진다.
  3. 실전 운용은 항상 '지금까지의 자료로 다음을 맞힌다' 이다. CV 가 그 상황과 같아야
     CV 점수가 운용 성과의 대리가 된다.

두 겹으로 쓴다 (nested CV).
  · 바깥(outer): 확장창 walk-forward. OOS 성과 평가용. 매 분기 모델을 그때까지 자료로 다시 적합.
  · 안쪽(inner): 그 시점 훈련창 **안에서** 다시 시간순 분할 → 하이퍼파라미터(λ 등) 선택.
    안쪽을 생략하고 전체 표본으로 λ 를 고르면, 그 λ 자체가 테스트 구간을 본 값이 되어 누수다.
"""
from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def expanding_splits(index: pd.DatetimeIndex, min_train: int, start: pd.Timestamp | None = None,
                     embargo: int = 0) -> Iterator[tuple[pd.DatetimeIndex, pd.Timestamp]]:
    """확장창 walk-forward 분할.

    각 스텝마다 (훈련 인덱스, 예측 시점 1개) 를 내놓는다.
      · 훈련 = index 의 처음부터 t 직전까지, 길이가 min_train 이상일 때만
      · embargo > 0 이면 t 직전 embargo 개를 훈련에서 뺀다 (De Prado 의 purging/embargo.
        분기 수익률은 겹치지 않으니 0 이 기본, 라벨이 겹치는 3단계에서 쓴다)
      · start 가 주어지면 그 시점 이후만 예측 대상

    TODO: for 루프 하나로 충분하다. yield 로 내놓는다.
    """
    raise NotImplementedError


def inner_splits(n: int, min_train: int) -> Iterator[tuple[slice, int]]:
    """훈련창 **안쪽** 시간순 CV — 하이퍼파라미터 선택용.

    t = min_train .. n−1 각각에 대해 (slice(0, t), t) 를 내놓는다. 즉 한 점씩 앞으로 밀며
    확장창으로 예측한다. factor-nowcasting 의 RidgeDL 이 λ 를 고르는 방식과 같다.

    TODO: 두 줄이면 된다.
    """
    raise NotImplementedError


def leakage_checklist(X: pd.DataFrame, y: pd.Series, train_idx: pd.DatetimeIndex,
                      predict_at: pd.Timestamp) -> dict[str, bool]:
    """결과가 좋을 때 먼저 돌리는 체크리스트. 전부 True 여야 한다.

    반환 키:
      no_future_train   : train_idx 의 모든 시점 < predict_at
      no_nan_in_design  : X.loc[train_idx] 와 y.loc[train_idx] 에 NaN 없음
      target_not_in_X   : X 열에 y 자신(동시점)이 섞여 있지 않음 — y_l1 은 허용, y_l0 는 금지
      no_duplicate_rows : train_idx 에 중복 시점 없음
      x_row_available   : predict_at 행의 X 가 결측 없이 존재

    TODO: 각 항목을 assert 가 아니라 bool 로 돌려준다 (리포트에 표로 찍기 위해).
          실패한 항목이 있으면 호출부에서 멈추게 한다.
    """
    raise NotImplementedError
