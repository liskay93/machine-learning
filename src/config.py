"""경로·시드. 원천 저장소는 읽기 전용이며 환경변수로 위치를 덮어쓸 수 있다."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def load_paths(path: str | Path = "config/paths.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def factor_nowcasting_root(paths: dict | None = None) -> Path:
    env = os.environ.get("FACTOR_NOWCASTING_ROOT")
    return Path(env).resolve() if env else resolve((paths or load_paths())["factor_nowcasting_root"])


def macro_factor_root(paths: dict | None = None) -> Path:
    env = os.environ.get("MACRO_FACTOR_ROOT")
    return Path(env).resolve() if env else resolve((paths or load_paths())["macro_factor_root"])
