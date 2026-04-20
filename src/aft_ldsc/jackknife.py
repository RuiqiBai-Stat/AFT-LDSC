from __future__ import annotations

import numpy as np


def _n_blocks_for_n(n: int, n_blocks: int) -> int:
    return max(2, min(int(n_blocks), int(n)))


def fast_block_jackknife(y: np.ndarray, x: np.ndarray, w: np.ndarray, n_blocks: int = 200) -> dict:
    """
    Fast delete-1-block jackknife using blockwise X'WX and X'Wy accumulators.
    """
    n = x.shape[0]
    k = x.shape[1]
    b = _n_blocks_for_n(n, n_blocks)
    boundaries = np.floor(np.linspace(0, n, b + 1)).astype(int)

    xtx_block = np.zeros((b, k, k), dtype=float)
    xty_block = np.zeros((b, k), dtype=float)

    for i in range(b):
        s, e = boundaries[i], boundaries[i + 1]
        xb = x[s:e]
        yb = y[s:e]
        wb = w[s:e]
        xtx_block[i] = (xb.T * wb) @ xb
        xty_block[i] = (xb.T * wb) @ yb

    xtx_total = xtx_block.sum(axis=0)
    xty_total = xty_block.sum(axis=0)
    try:
        full = np.linalg.solve(xtx_total, xty_total)
    except np.linalg.LinAlgError:
        full = np.linalg.pinv(xtx_total) @ xty_total

    delete_vals = np.zeros((b, k), dtype=float)
    for i in range(b):
        xtx_del = xtx_total - xtx_block[i]
        xty_del = xty_total - xty_block[i]
        try:
            delete_vals[i] = np.linalg.solve(xtx_del, xty_del)
        except np.linalg.LinAlgError:
            delete_vals[i] = np.linalg.pinv(xtx_del) @ xty_del

    pseudo = b * full - (b - 1) * delete_vals
    cov = np.cov(pseudo.T, ddof=1) / b
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    return {
        "full": full,
        "delete": delete_vals,
        "cov": cov,
        "se": se,
        "n_blocks": b,
    }


def jackknife_se_scalar(full_value: float, delete_values: np.ndarray) -> float:
    delete_values = np.asarray(delete_values, dtype=float)
    b = len(delete_values)
    pseudo = b * full_value - (b - 1) * delete_values
    var = np.var(pseudo, ddof=1) / b
    return float(np.sqrt(max(var, 0.0)))
