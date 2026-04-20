from __future__ import annotations

import numpy as np


def _safe_inverse(a: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(a)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a)


def weighted_lstsq(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xtwx = (x.T * w) @ x
    xtwy = (x.T * w) @ y
    beta = _safe_inverse(xtwx) @ xtwy
    return beta


def estimate_h2_iterative(
    y: np.ndarray,
    ld: np.ndarray,
    nbar: float,
    alpha: float,
    m_snp: int,
    weight_ld: np.ndarray | None = None,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> dict:
    """
    y: V^2
    model: y = intercept + beta * (ld * alpha^2)
    h2 = beta * M / N
    """
    ld = np.asarray(ld, dtype=float)
    y = np.asarray(y, dtype=float)
    x = np.column_stack([np.ones_like(ld), ld * (alpha**2)])
    wld = np.maximum(weight_ld if weight_ld is not None else ld, 1.0)

    num = np.mean((y - alpha) * m_snp)
    den = np.mean(ld * nbar * (alpha**2))
    h2_init = np.clip(num / max(den, 1e-12), 0.0, 1.0)

    expected = alpha + (h2_init * nbar * (alpha**2) / max(m_snp, 1)) * ld
    expected = np.clip(expected, 1e-12, None)
    w = 1.0 / (2.0 * (expected**2) * wld)
    beta = weighted_lstsq(y, x, w)

    intercept_old = beta[0]
    h2_old = beta[1] * m_snp / nbar
    it = 0
    for it in range(1, max_iter + 1):
        expected = intercept_old + (h2_old * nbar * (alpha**2) / max(m_snp, 1)) * ld
        expected = np.clip(expected, 1e-12, None)
        w = 1.0 / (2.0 * (expected**2) * wld)
        beta = weighted_lstsq(y, x, w)
        intercept_new = beta[0]
        h2_new = beta[1] * m_snp / nbar
        if abs(intercept_new - intercept_old) < tol and abs(h2_new - h2_old) < tol:
            intercept_old = intercept_new
            h2_old = h2_new
            break
        intercept_old = intercept_new
        h2_old = h2_new

    return {
        "beta": beta,
        "weights": w,
        "h2": float(h2_old),
        "intercept": float(intercept_old),
        "iterations": it,
        "x": x,
    }


def estimate_partition_coefficients(
    y: np.ndarray,
    ld_mat: np.ndarray,
    nbar: float,
    alpha: float,
    weight_ld: np.ndarray,
) -> dict:
    """
    Minimal partitioned regression following baselineLD_survival style:
    - one weighted fit
    - divide coefficients by Nbar to get tau scale
    """
    y = np.asarray(y, dtype=float)
    ld_mat = np.asarray(ld_mat, dtype=float)
    ld_sum = ld_mat.sum(axis=1)
    tau_init = (np.mean(y) - alpha) / max(nbar * (alpha**2) * np.mean(ld_sum), 1e-12)
    w = 1.0 / (np.maximum(weight_ld, 1.0) * np.clip(alpha + nbar * tau_init * (alpha**2) * ld_sum, 1e-12, None) ** 2)
    x = np.column_stack([np.ones(len(y)), ld_mat * (alpha**2)])
    beta_raw = weighted_lstsq(y, x, w)
    beta = beta_raw / nbar
    return {"beta_raw": beta_raw, "beta": beta, "weights": w, "x": x}


def estimate_gcov_iterative(
    zprod: np.ndarray,
    ld: np.ndarray,
    n1: float,
    n2: float,
    h2_1: float,
    h2_2: float,
    intercept_1: float,
    intercept_2: float,
    alpha1: float,
    alpha2: float,
    m_snp: int,
    weight_ld: np.ndarray | None = None,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> dict:
    """
    model: E[z1*z2] = intercept + beta * (ld * alpha1 * alpha2)
    gcov = beta * M / sqrt(N1*N2)
    """
    zprod = np.asarray(zprod, dtype=float)
    ld = np.asarray(ld, dtype=float)
    n_cross = float(np.sqrt(n1 * n2))
    wld = np.maximum(weight_ld if weight_ld is not None else ld, 1.0)

    x = np.column_stack([np.ones_like(ld), ld * alpha1 * alpha2])

    num = np.mean(zprod * m_snp)
    den = np.mean(ld * n_cross * alpha1 * alpha2)
    gcov_init = num / max(den, 1e-12)

    a = (n1 * h2_1 * (alpha1**2) / max(m_snp, 1)) * ld + intercept_1
    b = (n2 * h2_2 * (alpha2**2) / max(m_snp, 1)) * ld + intercept_2
    c = (n_cross * gcov_init * alpha1 * alpha2 / max(m_snp, 1)) * ld
    w = 1.0 / (np.clip(a * b + c**2, 1e-12, None) * wld)
    beta = weighted_lstsq(zprod, x, w)

    intercept_old = beta[0]
    gcov_old = beta[1] * m_snp / n_cross
    it = 0
    for it in range(1, max_iter + 1):
        c = (n_cross * gcov_old * alpha1 * alpha2 / max(m_snp, 1)) * ld + intercept_old
        w = 1.0 / (np.clip(a * b + c**2, 1e-12, None) * wld)
        beta = weighted_lstsq(zprod, x, w)
        intercept_new = beta[0]
        gcov_new = beta[1] * m_snp / n_cross
        if abs(intercept_new - intercept_old) < tol and abs(gcov_new - gcov_old) < tol:
            intercept_old = intercept_new
            gcov_old = gcov_new
            break
        intercept_old = intercept_new
        gcov_old = gcov_new

    return {
        "beta": beta,
        "weights": w,
        "gcov": float(gcov_old),
        "intercept": float(intercept_old),
        "x": x,
        "iterations": it,
    }
