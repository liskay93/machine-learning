"""평가 지표 — 예측이 얼마나 맞았는지 재는 자(尺).

이 파일이 하는 일은 하나입니다. **예측값과 실제값을 넣으면 점수 하나를 돌려주는 함수들.**
모델을 만들기 전에 먼저 만드는 이유는, 무엇으로 이겼다고 할지 미리 정해 두지 않으면
나중에 자기에게 유리한 지표를 골라 버리기 때문입니다.

아래 예시를 머리에 넣고 읽으시면 됩니다. 어느 분기의 실제 수익률과 예측이 이랬다고 칩시다.

    분기          2024Q1   2024Q2   2024Q3
    실제 y         +3.0%    -1.0%    +2.0%
    내 예측 yhat   +2.0%    +0.5%    +2.5%
    오차 e=y-yhat  +1.0%p   -1.5%p   -0.5%p

이 세 오차를 어떻게 한 숫자로 요약하느냐가 지표입니다.

`raise NotImplementedError` 는 "아직 안 짰다"는 표시입니다. 그 줄을 지우고 실제 계산을
써 넣으면 됩니다. rmse 와 mae 는 예시로 채워 뒀으니 그걸 보고 나머지를 같은 식으로 채우시면 됩니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Root Mean Squared Error — 오차를 제곱해 평균 내고 다시 제곱근.

    위 예시: 오차 (1.0, -1.5, -0.5)%p
        제곱      → (1.00, 2.25, 0.25)
        평균      → 1.1667
        제곱근    → 1.08%p

    제곱을 하므로 **큰 오차 하나가 작은 오차 여럿보다 훨씬 아프게** 잡힙니다.
    단위가 원래 값과 같아서(%p) 그대로 읽을 수 있습니다. "이 모델은 분기 수익률을
    평균 1.08%p 틀린다"로 해석하시면 됩니다. 이 저장소의 기본 지표입니다.
    """
    e = np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean(e ** 2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean Absolute Error — 오차의 절댓값 평균.

    위 예시: |1.0|, |1.5|, |0.5| → 평균 1.00%p

    제곱을 안 하니 큰 오차를 덜 벌합니다. **rmse 와 mae 가 크게 벌어지면**
    (예: rmse 3.5%p 인데 mae 1.2%p) 몇 개 분기의 대형 오차가 성적을 지배한다는 뜻입니다.
    보통 2008년이나 2020년 같은 위기 분기입니다. 둘을 같이 보는 이유가 이것입니다.
    """
    e = np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(e)))


def hit_ratio(y: np.ndarray, yhat: np.ndarray) -> float:
    """부호 적중률 — 오르내림 방향을 맞힌 비율.

    위 예시: 실제 (+,−,+) vs 예측 (+,+,+) → 1·3번만 맞음 → 2/3 = 0.67

    크기는 틀려도 방향만 맞으면 되는 의사결정(오버웨이트냐 언더웨이트냐)에서는
    rmse 보다 이쪽이 맞는 자입니다. 단, 대체투자는 플러스 분기가 압도적으로 많아
    "항상 +" 라고 찍어도 0.85 가 나옵니다. 그래서 **"항상 +" 대비** 로 읽어야 합니다.

    TODO
      1. np.sign(y) 와 np.sign(yhat) 이 같은 비율을 구한다 (np.mean 으로 평균 내면 비율).
      2. y 가 정확히 0 인 분기를 어떻게 셀지 정하고 이 docstring 에 한 줄로 적는다.
         (정답은 없습니다. 제외해도 되고 틀린 것으로 쳐도 됩니다. 정하고 적어 두는 게 핵심)
    """
    raise NotImplementedError


def r2_oos(y: np.ndarray, yhat: np.ndarray, y_bench: np.ndarray) -> float:
    """표본 밖 R² (Campbell-Thompson) — **벤치마크보다 얼마나 나은지**.

        r2_oos = 1 − (내 모델 오차 제곱합) / (벤치마크 오차 제곱합)

    읽는 법
        0.6  → 벤치마크 대비 오차 제곱합을 60% 줄였다. 좋다.
        0.0  → 벤치마크와 똑같다. 모델을 만든 의미가 없다.
        음수 → 벤치마크보다 **나쁘다**. 실제로 자주 나옵니다.

    보통 R² 와 다른 점: 학교에서 배우는 R² 는 "평균 대비 설명력"을 같은 표본 안에서 잽니다.
    이것은 **학습에 안 쓴 구간에서, 내가 고른 벤치마크 대비** 로 잽니다. 훨씬 엄격합니다.

    ⚠ 벤치마크가 무엇인지 반드시 같이 적습니다. 같은 모델이라도
      "확장창 평균 대비 0.14" 와 "AR(1) 대비 −0.40" 이 동시에 성립합니다. 전혀 다른 이야기입니다.

    TODO: 분자는 sum((y−yhat)²), 분모는 sum((y−y_bench)²). 나눠서 1에서 뺀다. 한 줄입니다.
    """
    raise NotImplementedError


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """두 모델의 오차 차이가 **운인지 실력인지** 보는 검정. (지금 당장은 안 짜도 됩니다)

    모델 A 의 RMSE 2.13%, 모델 B 가 2.19% 라고 해서 A 가 낫다고 할 수 있을까요?
    57분기밖에 없으면 그 차이가 우연일 수 있습니다. 그걸 보는 것이 이 검정입니다.

    d_t = e1_t² − e2_t² 를 만들고, 그 평균이 0 인지 t 검정합니다.
    통계량이 음수면 모델 1 이 낫고, p 값이 작으면 (보통 0.05 미만) 우연으로 보기 어렵습니다.

    ⚠ 표본이 34~57분기로 짧아 검정력이 낮습니다. **참고용**이지 이걸로 결론 내지 않습니다.

    TODO (나중에)
      1. d = e1**2 - e2**2, dbar = d.mean()
      2. 분산: gamma0 = sum((d-dbar)**2)/n. h>1 이면 자기공분산 항을 더한다 (HAC)
      3. dm = dbar / sqrt(var/n), p = 양측 t 검정. var <= 0 이면 (np.nan, np.nan) 반환
    """
    raise NotImplementedError


def summarize(y: pd.Series, preds: pd.DataFrame, bench: str = "mean", dm_vs: str = "ols") -> pd.DataFrame:
    """여러 모델을 한 표로 묶는다. 위 함수들을 엮기만 하는 껍데기입니다.

    preds 는 이렇게 생긴 프레임입니다 (행 = 예측 분기, 열 = 모델).

                    mean     ar1     ols   ridge
        2012-03-31  0.027   0.029   0.087   0.062
        2012-06-30  0.028   0.040  -0.001   0.007
        ...

    반환은 행 = 모델, 열 = [n, rmse, mae, hit, r2_oos_vs_<bench>] 인 표.

    TODO
      1. for 문으로 preds 의 각 열을 돌면서 위 지표 함수들을 호출해 dict 에 담는다
      2. r2_oos 의 y_bench 로는 preds[bench] 열을 쓴다
      3. pd.DataFrame(rows).T 로 모아 반환한다
    """
    raise NotImplementedError
