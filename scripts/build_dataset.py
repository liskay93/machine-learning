"""분기 패널 빌드 — 읽기 전용 원천 두 곳에서 읽어 data/processed/panel_quarterly.csv 로 저장한다.

    factor-nowcasting : data/raw/alt_returns_bm.xlsx (대체투자 15 시리즈 분기 단순수익률), config/nowcast.yaml (라벨→시리즈 매핑)
    macro_factor      : data/processed/factors.csv (4 팩터 일간 단순수익률)

집계 규약은 factor-nowcasting 의 altnow 패키지를 그대로 쓴다 (일간 → 분기 복리, 부분 분기 제외).
이 스크립트는 데이터 배관이라 완성본을 준다. 모델링 코드는 src/ 의 뼈대를 직접 채운다.

    FACTOR_NOWCASTING_ROOT=... MACRO_FACTOR_ROOT=... python scripts/build_dataset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import factor_nowcasting_root, load_paths, macro_factor_root, resolve  # noqa: E402


def main() -> int:
    paths = load_paths()
    fn_root, mf_root = factor_nowcasting_root(paths), macro_factor_root(paths)
    for r in (fn_root, mf_root):
        if not r.exists():
            raise FileNotFoundError(f"원천 저장소 없음: {r} (config/paths.yaml 또는 환경변수 확인)")
    # altnow 는 factor-nowcasting/src 아래 패키지. 이 저장소의 src 와 이름이 겹치지 않게 그 디렉터리를 직접 넣는다.
    sys.path.insert(0, str(fn_root / "src"))
    import os
    os.environ["MACRO_FACTOR_ROOT"] = str(mf_root)
    from altnow.backtest import FactorPanels
    from altnow.config import load_config
    from altnow.data import load_alt_returns, load_factors_daily

    cfg = load_config(fn_root / "config" / "nowcast.yaml")
    alt = load_alt_returns(cfg)
    f_daily = load_factors_daily(cfg)
    panels = FactorPanels.from_daily(f_daily)

    fq = panels.fq_full                                   # 완전 분기만 (부분 분기 2006Q1·진행 중 분기 제외)
    # 대체투자 이력은 2000Q1 부터라 팩터보다 길다. 잘라내지 않는다 — y_{q-1} 시차와
    # 확장창 평균 베이스라인이 팩터 이전 구간을 쓰기 때문이다. 팩터는 그 구간에서 NaN 으로 남는다.
    panel = pd.concat([fq, alt.series], axis=1, sort=True)
    panel.index.name = "date"

    out = resolve(paths["processed"]); out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out / "panel_quarterly.csv", float_format="%.10g")
    meta = alt.meta.copy(); meta["labels"] = meta["labels"].apply(lambda l: "|".join(l))
    meta.to_csv(out / "series_meta.csv")
    info = {"factors": list(fq.columns), "series": list(alt.series.columns),
            "first_quarter": str(panel.index[0].date()), "first_factor_quarter": str(fq.index[0].date()),
            "last_factor_quarter": str(fq.index[-1].date()),
            "last_alt_quarter": str(alt.series.dropna(how="all").index[-1].date()),
            "n_quarters": int(len(panel)), "factor_asof": str(f_daily.index[-1].date()),
            "source_factor_nowcasting": str(fn_root), "source_macro_factor": str(mf_root)}
    (out / "panel_meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(panel.tail(3).round(4).to_string())
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
