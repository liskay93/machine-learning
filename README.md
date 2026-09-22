# machine-learning — 금융 ML 학습 저장소

자산배분 실무에서 쓰는 문제로 머신러닝을 순서대로 익힌다. 목표는 "돌아가는 코드"가 아니라
**왜 그 모델이 그렇게 행동하는지** 를 아는 것이다.

## 로드맵

| 단계 | 주제 | 배우는 것 | 상태 |
|---|---|---|---|
| 1 | 팩터 nowcasting: OLS → Ridge/Lasso/ElasticNet → LightGBM | 정규화, 과적합, 하이퍼파라미터 튜닝 | 진행 중 |
| 2 | Preqin 캐시플로로 PE 콜/분배 예측 (Takahashi-Alexander vs LightGBM) | walk-forward CV, SHAP | 대기 |
| 3 | CTA 추세 신호 메타 라벨링 | 분류, 확률 보정, 클래스 불균형, 누수 방지 | 대기 |
| 4 | 레짐 분석: HMM · GMM · k-means · 변화점 탐지 | 비지도 모델 비교와 평가 | 대기 |
| 5 | GP 분기 보고서 PDF 구조화 추출 | 임베딩, RAG, 평가셋 | 대기 |

단계별 정리는 `notes/step_N.md` 에 쌓는다.

## 규칙

- **모델링 순서**: 단순 베이스라인 → 지표 정의 → ML 모델 → 베이스라인 대비 비교 → 언제 지는지 분석.
- **시계열 CV 만** 쓴다. 무작위 KFold 금지 (`src/cv.py` 참고).
- **결과가 좋으면 먼저 의심한다**: 누수 · 룩어헤드 · 중복 샘플 · 테스트셋 오염. `src/cv.leakage_checklist` 를 돌리고 결과 보고에 명시한다.
- 노트북에 로직을 쌓지 않는다. 검증된 코드는 `src/` 로 옮기고 테스트를 붙인다.
- 재현성: 시드 고정 (`src.config.SEED`), `requirements.txt` 유지.

## 데이터 — 원천 저장소는 읽기 전용

이 저장소는 데이터를 만들지 않는다. 두 저장소에서 **읽기만** 한다.

| 원천 | 쓰는 것 | 경로 지정 |
|---|---|---|
| [`liskay93/factor-nowcasting`](https://github.com/liskay93/factor-nowcasting) | 대체투자 15 시리즈 분기 수익률 (`data/raw/alt_returns_bm.xlsx`), 라벨 매핑, 집계 규약(`altnow`) | `FACTOR_NOWCASTING_ROOT` |
| [`liskay93/macro_factor`](https://github.com/liskay93/macro_factor) | 매크로 팩터 4종 일간 수익률 (`data/processed/factors.csv`) | `MACRO_FACTOR_ROOT` |

기본 경로는 `config/paths.yaml` (형제 디렉터리 가정), 환경변수가 있으면 그것을 우선한다.

```bash
pip install -r requirements.txt
FACTOR_NOWCASTING_ROOT=../factor-nowcasting MACRO_FACTOR_ROOT=../macro_factor \
  python scripts/build_dataset.py          # → data/processed/panel_quarterly.csv
python -m pytest -q
```

`data/raw/` 는 git 에서 제외한다 (원천 저장소가 소유). `data/processed/` 의 빌드 산출물은 커밋한다.

### 팩터 부호 규약

`term` + = 금리 하락 · `inflation` + = 기대인플레(BEI) 상승 · `credit` + = 스프레드 축소 · `growth` + = 주가 상승.
부호를 반대로 읽으면 계수 해석이 통째로 뒤집힌다. 자세한 정의는 `macro_factor/SPEC.md`.

## 모델이 좋은지 어떻게 판정하는가 (모든 단계 공통)

ML 모델의 상대는 **과거 평균이 아니라 정규화 없는 OLS** 다. 여기를 틀리면 판정 전체가 무너진다.
과거 평균을 이겼다는 것은 "팩터가 쓸모 있다" 까지만 말한 것이지 "ML 이 쓸모 있다" 가 아니다.

관문을 순서대로 통과시킨다. 하나라도 막히면 거기서 멈춘다.

| 관문 | 묻는 것 | 통과 조건 |
|---|---|---|
| 0 · 누수 | 결과가 좋으면 **먼저 의심** | `cv.leakage_checklist` 5개 항목 전부 True |
| 1 · 과거 평균 | 팩터가 쓸모 있나 | r2_oos(대 평균) > 0 |
| 2 · AR(1) | 자기 관성 이상을 잡았나 | RMSE < AR(1) 의 RMSE |
| 3 · **OLS** | **ML 이 값어치가 있나** | RMSE < 정규화 없는 OLS 의 RMSE |
| 4 · 운 | 그 차이가 표본 운인가 | DM p < 0.05 (표본이 짧아 참고용) |
| 5 · 경제 | 계수 부호가 말이 되나 | growth 베타 음수 같은 것이 없다 |
| 6 · 실무 | 의사결정이 바뀌나 | RMSE 개선이 배분 판단을 바꿀 크기인가 |

### 실측 (2026-09-22, AR 항 없음, OOS 2012Q1~, 56분기, r2_oos 는 확장창 평균 대비)

| 시리즈 | AR(1) | OLS lag0 | OLS lag0~4 | Ridge lag0~4 | Lasso lag0~4 | Lasso 생존변수 | DM p (Lasso vs OLS) |
|---|---|---|---|---|---|---|---|
| buyout | 0.04 | 0.53 | 0.24 | 0.54 | **0.59** | 6.6 / 20 | 0.31 |
| hfri_fof_cons | −0.11 | **0.72** | 0.14 | 0.68 | 0.67 | 8.6 / 20 | 0.42 |
| epra_nareit | −0.08 | **0.55** | 0.09 | 0.54 | 0.50 | 6.1 / 20 | 0.58 |
| infra_core | −0.05 | 0.14 | −1.64 | −0.02 | **0.21** | 1.9 / 20 | 0.71 |
| re_noncore | **0.40** | −0.93 | −5.45 | −1.21 | −0.88 | 9.6 / 20 | 0.91 |

**읽는 법**

- 관문 3 에서 ML 이 OLS 를 이긴 것은 5개 중 2개(buyout, infra_core)뿐이고, 나머지는 단순 OLS 가 낫다.
- 관문 4 에서 **전부 탈락한다.** DM p 값이 모두 0.3 이상이라 그 2개의 개선조차 운과 구분되지 않는다.
  56분기는 이 크기의 차이를 가려낼 만큼 길지 않다.
- OLS lag0~4 열이 정규화의 존재 이유다. 변수를 20개로 늘리면 성적이 무너지고(infra_core −1.64),
  Ridge·Lasso 가 그 피해를 되돌린다. **정규화는 망가진 모델을 복구할 뿐 없는 신호를 만들지 못한다.**
- Lasso 가 20개 중 6~9개만 남긴다. infra_core 는 1.9개 — 거의 전부 죽인다. 시차 팩터에 정보가 없다는
  뜻이고, 원본 factor-nowcasting 이 "A 의 lag 1~4 는 OOS 에서 도움이 안 된다" 고 한 결론과 같다.
- re_noncore 는 모든 팩터 모델이 과거 평균에도 진다. AR(1) 만 0.40 이다. 감정평가 지수라
  작년 평가액이 올해를 결정한다 — 팩터가 아니라 관성이 움직인다.

**결론: 이 데이터에서 1단계 ML 은 OLS 를 이기지 못했다.** 이것이 실패가 아니라 정상적인 결과다.
변수 20개에 표본 56개면 정규화로 얻을 것이 거의 없다. 얻을 것이 생기려면 변수가 훨씬 많거나
(2단계 캐시플로 특징량) 관계가 비선형이어야 한다 (LightGBM).

### 비교 가능성 제약 — AR 항을 쓰지 않는다

이 저장소의 nowcast 결과는 공모자산(주식·채권)과 **같은 표에 놓고 비교**해야 한다.
그래서 설계행렬에 직전 분기 수익률 y_{q−1} 을 넣지 않는다. 공모자산 모델은 `r = a + b'F` 뿐인데
대체투자에만 자기 과거를 넣으면 팩터가 설명할 몫을 AR 항이 먼저 가져가 베타가 깎인다.

AR 항을 뺀 비용 (buyout 외 4개, OOS 2012Q1~, R²_OOS 는 확장창 평균 대비):

| 시리즈 | θ(스무딩) | AR(1) 단독 | 팩터+AR | **팩터만** |
|---|---|---|---|---|
| hfri_fof_cons | 0.09 | −0.12 | 0.73 | **0.72** |
| epra_nareit | 0.07 | −0.07 | 0.56 | **0.57** |
| buyout | 0.30 | 0.04 | 0.67 | **0.53** |
| msci_gpfi | 0.71 | 0.33 | 0.24 | **−0.47** |
| re_noncore | 0.63 | 0.40 | 0.14 | **−0.93** |

비용이 θ 에 비례한다. 상장 부동산·헤지펀드처럼 스무딩이 없는(θ≈0.1) 시리즈는 잃는 것이 없고,
감정평가 부동산(θ 0.6~0.7)은 음수로 떨어진다 — 보고된 값이 팩터가 아니라 자기 관성으로 움직인다는 뜻이다.

### ⚠ 베타 감쇠 — 보고 수익률에 그냥 회귀하면 베타가 작게 나온다

AR 항을 빼도 비교 가능성 문제가 다 풀리지는 않는다. 감정평가 스무딩이 있으면
보고 수익률에 대한 회귀 계수는 **경제적 베타 × (1−θ)** 로 줄어든 값이다.

| 시리즈 | θ | 보고 베타(growth) | 경제 베타 = 보고/(1−θ) |
|---|---|---|---|
| msci_gpfi | 0.71 | 0.30 | 1.07 |
| re_noncore | 0.63 | 0.55 | 1.47 |
| vc | 0.50 | 0.58 | 1.16 |
| buyout | 0.30 | 0.53 | 0.75 |
| epra_nareit | 0.07 | 0.88 | 0.94 |

코어 부동산의 주식 베타를 0.30 으로 읽고 공모주식 1.0 과 비교하면 노출을 3배 이상 과소평가한다.
**따라서 베타를 공모자산과 직접 비교하려면 언스무딩이 한 겹 더 필요하다** (Geltner AR(p), 또는
원본 저장소의 상태공간 모델이 θ 와 β 를 같이 추정하는 방식). 1단계는 정규화 학습에 집중하므로
보고 수익률을 그대로 타깃으로 쓰고, 언스무딩은 별도 단계로 뗀다. θ 는 위 표를 잠정값으로 쓴다
(출처: `factor-nowcasting/data/processed/model_b_params_full.csv`).

### 빌드 검증 (2026-09-22)

- 패널의 `y` 가 `factor-nowcasting` 백테스트 파일의 `y` 와 일치 (최대 오차 4.9e-9 = CSV 반올림).
- walk-forward 동분기 OLS 를 독립 구현해 원본 `ols` 열을 재현 (최대 오차 4.9e-9).
- 일간 → 분기 복리를 `altnow` 없이 다시 계산해 일치 (최대 오차 5.0e-11), 부분 분기 판정도 동일 (2006Q1 · 2026Q3 제외).
- 상식 확인: `growth` 연간 수익률 2008년 −42.2%, 2020년 +16.3% (MSCI ACWI 실적과 부합).

## 구조

```
config/paths.yaml     원천 저장소 경로
config/step1.yaml     1단계 설정 (대상 시리즈·피처·CV·하이퍼파라미터 격자)
scripts/              데이터 빌드 등 배관
src/config.py         경로·시드
src/data.py           패널 적재
src/features.py       설계행렬·표준화          (뼈대)
src/cv.py             시계열 CV·누수 체크리스트  (뼈대)
src/baselines.py      평균·AR(1)·OLS           (뼈대)
src/metrics.py        RMSE·MAE·적중률·R²_OOS·DM (뼈대)
src/models_linear.py  Ridge·Lasso·ElasticNet   (뼈대)
notebooks/            탐색용. 검증되면 src/ 로 옮긴다
notes/step_N.md       단계별 배운 것 / 실무 적용 가능성 / 한계
tests/                정합성 검사
```
