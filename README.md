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
