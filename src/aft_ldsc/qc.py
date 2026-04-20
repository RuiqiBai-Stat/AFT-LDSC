from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def assert_file_exists(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"File does not exist: {path}")


def require_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def drop_nonfinite(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for c in cols:
        mask &= np.isfinite(pd.to_numeric(df[c], errors="coerce"))
    return df.loc[mask].copy()


def validate_alpha(alpha: float) -> float:
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite.")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}.")
    return float(alpha)
