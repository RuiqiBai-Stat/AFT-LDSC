from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aft_ldsc.io_utils import _resolve_with_compression, get_present_chrs, read_alpha, read_ldsc_table, split_csv_arg, sub_chr, write_table
from aft_ldsc.qc import require_columns, validate_alpha


def _read_chr_table(prefix: str, chrom: int, suffix: str, sep: str = r"\s+") -> pd.DataFrame:
    base = sub_chr(prefix, chrom) + suffix
    path, comp = _resolve_with_compression(base)
    return pd.read_csv(path, sep=sep, compression=comp)


def _detect_model(ref_prefix: str) -> tuple[str, list[str]]:
    chrs = get_present_chrs(ref_prefix, max_chr=22)
    if not chrs:
        raise ValueError(f"No chromosome files found for --ref-ld-chr {ref_prefix}")
    first = _read_chr_table(ref_prefix, chrs[0], ".l2.ldscore")
    ld_cols = [c for c in first.columns if c not in {"CHR", "SNP", "BP", "CM", "MAF"}]
    if len(ld_cols) <= 1:
        return "ld", ld_cols
    return "baselineLD", ld_cols


def _weighted_solve(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xtx = (x.T * w) @ x
    xty = (x.T * w) @ y
    try:
        return np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx) @ xty


def _jackknife_scaled(y: np.ndarray, x: np.ndarray, w: np.ndarray, n_blocks: int, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = x.shape[0]
    k = x.shape[1]
    bounds = np.floor(np.linspace(0, m, n_blocks + 1)).astype(int)
    xtx_block = np.zeros((n_blocks, k, k))
    xty_block = np.zeros((n_blocks, k))
    for i in range(n_blocks):
        s, e = bounds[i], bounds[i + 1]
        xb, yb, wb = x[s:e], y[s:e], w[s:e]
        xtx_block[i] = (xb.T * wb) @ xb
        xty_block[i] = (xb.T * wb) @ yb
    xtx_total = xtx_block.sum(axis=0)
    xty_total = xty_block.sum(axis=0)
    try:
        full = np.linalg.solve(xtx_total, xty_total) / scale
    except np.linalg.LinAlgError:
        full = (np.linalg.pinv(xtx_total) @ xty_total) / scale
    delete = np.zeros((n_blocks, k))
    for i in range(n_blocks):
        xtx_del = xtx_total - xtx_block[i]
        xty_del = xty_total - xty_block[i]
        try:
            delete[i] = np.linalg.solve(xtx_del, xty_del) / scale
        except np.linalg.LinAlgError:
            delete[i] = (np.linalg.pinv(xtx_del) @ xty_del) / scale
    pseudo = n_blocks * full - (n_blocks - 1) * delete
    cov = np.cov(pseudo.T, ddof=1) / n_blocks
    return full, delete, cov


def _load_m_total(ref_prefix: str, chrs: list[int]) -> float | None:
    vals = []
    for chrom in chrs:
        base = sub_chr(ref_prefix, chrom) + ".l2.M_5_50"
        try:
            with open(base, "r", encoding="utf-8") as f:
                row = f.readline().strip().split()
            if row:
                vals.append(float(row[0]))
        except FileNotFoundError:
            return None
    if not vals:
        return None
    return float(np.sum(vals))


def _compute_baseline_mode(
    sumstats: pd.DataFrame,
    alpha: float,
    ref_prefix: str,
    w_prefix: str,
    frq_prefix: str | None,
    n_blocks: int,
) -> dict:
    chrs = get_present_chrs(ref_prefix, max_chr=22)
    if not chrs:
        raise ValueError(f"No chromosome files found for --ref-ld-chr {ref_prefix}")

    ldscore_list = []
    weight_list = []
    z_list = []
    n_list = []
    annot_list = []

    for chrom in chrs:
        baseline_ld = _read_chr_table(ref_prefix, chrom, ".l2.ldscore", sep="\t")
        hm3_ld = _read_chr_table(w_prefix, chrom, ".l2.ldscore", sep="\t")
        require_columns(baseline_ld, ["SNP"], f"ref ld chr{chrom}")
        require_columns(hm3_ld, ["SNP", "L2"], f"w ld chr{chrom}")

        overlap = np.intersect1d(hm3_ld["SNP"].values, np.intersect1d(baseline_ld["SNP"].values, sumstats["SNP"].values))
        ldscore_list.append(baseline_ld[baseline_ld["SNP"].isin(overlap)].iloc[:, 3:].values)
        weight_list.append(hm3_ld[hm3_ld["SNP"].isin(overlap)]["L2"].values)
        z_list.append(sumstats[sumstats["SNP"].isin(overlap)]["Z"].values)
        n_list.append(sumstats[sumstats["SNP"].isin(overlap)]["N"].values)

        if frq_prefix is None:
            raise ValueError("baselineLD mode requires --frqfile-chr for annotation filtering.")
        annot_chr = _read_chr_table(ref_prefix, chrom, ".annot", sep="\t")
        frq = _read_chr_table(frq_prefix, chrom, ".frq", sep=r"\s+")
        if "MAF" in frq.columns:
            mask = frq["MAF"].values >= 0.05
        elif "FRQ" in frq.columns:
            mask = (frq["FRQ"].values >= 0.05) & (frq["FRQ"].values <= 0.95)
        else:
            raise ValueError(f"frq chr{chrom} missing MAF/FRQ column.")
        annot_list.append(annot_chr.iloc[np.where(mask)[0], 4:].values)

    ldscore = np.vstack(ldscore_list)
    weight = np.concatenate(weight_list).astype(float)
    weight[weight < 1] = 1
    z = np.concatenate(z_list).astype(float)
    y = z**2
    n_vec = np.concatenate(n_list).astype(float).reshape(-1, 1)
    annot = np.vstack(annot_list)
    annot_sum = np.sum(annot, axis=0)

    if len(y) == 0:
        raise ValueError("No SNPs left after overlap for baselineLD mode.")

    nbar = float(np.mean(n_vec).astype(int))
    ldscore = ldscore * n_vec / nbar

    x = np.column_stack([np.ones(len(y)), ldscore * (alpha**2)])
    tau_hat = (np.mean(y) - alpha) / (nbar * np.mean(np.sum(ldscore, axis=1)) * (alpha**2))
    w = 1.0 / weight / (alpha + nbar * tau_hat * np.sum(ldscore, axis=1) * (alpha**2)) ** 2

    full, delete, cov = _jackknife_scaled(y=y, x=x, w=w, n_blocks=n_blocks, scale=nbar)
    se = np.sqrt(np.diag(cov))

    intercept = float(full[0] * nbar / alpha)
    intercept_se = float(se[0] * nbar / alpha)
    total_h2 = float(annot_sum @ full[1:])
    total_h2_se = float(np.sqrt(annot_sum @ cov[1:, 1:] @ annot_sum.T))

    return {
        "mode": "baselineLD",
        "alpha": alpha,
        "n_snps": int(len(y)),
        "nbar": nbar,
        "intercept": intercept,
        "intercept_se": intercept_se,
        "heritability": total_h2,
        "heritability_se": total_h2_se,
    }


def _compute_ld_mode(
    sumstats: pd.DataFrame,
    alpha: float,
    ref_prefix: str,
    n_blocks: int,
    m_override: int | None = None,
    return_jackknife: bool = False,
) -> dict:
    chrs = get_present_chrs(ref_prefix, max_chr=22)
    if not chrs:
        raise ValueError(f"No chromosome files found for --ref-ld-chr {ref_prefix}")

    ldscore_list = []
    z_list = []
    n_list = []

    for chrom in chrs:
        ld_chr = _read_chr_table(ref_prefix, chrom, ".l2.ldscore")
        require_columns(ld_chr, ["SNP", "L2"], f"ld chr{chrom}")
        overlap = np.intersect1d(ld_chr["SNP"].values, sumstats["SNP"].values)
        ldscore_list.append(ld_chr[ld_chr["SNP"].isin(overlap)]["L2"].values)
        z_list.append(sumstats[sumstats["SNP"].isin(overlap)]["Z"].values)
        n_list.append(sumstats[sumstats["SNP"].isin(overlap)]["N"].values)

    ldscore = np.concatenate(ldscore_list).astype(float)
    z = np.concatenate(z_list).astype(float)
    y = z**2
    n_vec = np.concatenate(n_list).astype(float)

    if len(y) == 0:
        raise ValueError("No SNPs left after overlap for ld mode.")

    nbar = float(np.mean(n_vec).astype(int))
    ldscore[ldscore < 1] = 1
    ldscore = ldscore * n_vec / nbar

    m_total = float(m_override) if m_override is not None else _load_m_total(ref_prefix, chrs)
    if m_total is None:
        m_total = float(len(y))

    init_h2 = np.mean((y - alpha) * m_total) / np.mean(ldscore * nbar * (alpha**2))
    init_h2 = float(np.clip(init_h2, 0.0, 1.0))
    init_w = 1.0 / (2.0 * (alpha + init_h2 * nbar * (alpha**2) / m_total * ldscore) ** 2) / ldscore
    x = np.column_stack([np.ones(len(y)), ldscore * (alpha**2)])
    beta = _weighted_solve(y=y, x=x, w=init_w)

    intercept_old = float(beta[0])
    h2_old = float(beta[1] * m_total / nbar)
    w = init_w
    for _ in range(1000):
        w = 1.0 / (2.0 * (intercept_old + h2_old * nbar * (alpha**2) / m_total * ldscore) ** 2) / ldscore
        beta = _weighted_solve(y=y, x=x, w=w)
        intercept_new = float(beta[0])
        h2_new = float(beta[1] * m_total / nbar)
        if abs(intercept_new - intercept_old) <= 1e-8:
            intercept_old = intercept_new
            h2_old = h2_new
            break
        intercept_old = intercept_new
        h2_old = h2_new

    full, delete, cov = _jackknife_scaled(y=y, x=x, w=w, n_blocks=n_blocks, scale=1.0)
    intercept = float(full[0])
    intercept_se = float(np.sqrt(cov[0, 0]))
    h2 = float(full[1] * m_total / nbar)
    h2_se = float(np.sqrt(cov[1, 1]) * m_total / nbar)

    result = {
        "mode": "ld",
        "alpha": alpha,
        "n_snps": int(len(y)),
        "nbar": nbar,
        "m_total": m_total,
        "intercept": intercept,
        "intercept_se": intercept_se,
        "heritability": h2,
        "heritability_se": h2_se,
    }
    if return_jackknife:
        result["h2_jk_delete"] = delete[:, 1] * m_total / nbar
        result["intercept_jk_delete"] = delete[:, 0]
    return result


def run_heritability(args) -> dict:
    sumstats = read_ldsc_table(args.sumstats)
    require_columns(sumstats, ["SNP", "N", "Z", "A1", "A2"], "sumstats")
    alpha = validate_alpha(read_alpha(args.alpha))

    ref_prefixes = split_csv_arg(args.ref_ld_chr)
    if len(ref_prefixes) != 1:
        raise ValueError("heritability currently expects exactly one --ref-ld-chr prefix.")
    ref_prefix = ref_prefixes[0]

    mode, _ = _detect_model(ref_prefix)
    n_blocks = int(args.n_blocks)
    if n_blocks <= 1:
        raise ValueError("--n-blocks must be > 1.")

    if mode == "baselineLD":
        if args.w_ld_chr is None:
            raise ValueError("baselineLD mode requires --w-ld-chr.")
        result = _compute_baseline_mode(
            sumstats=sumstats,
            alpha=alpha,
            ref_prefix=ref_prefix,
            w_prefix=args.w_ld_chr,
            frq_prefix=args.frqfile_chr,
            n_blocks=n_blocks,
        )
    else:
        result = _compute_ld_mode(
            sumstats=sumstats,
            alpha=alpha,
            ref_prefix=ref_prefix,
            n_blocks=n_blocks,
            m_override=args.m,
        )

    result_df = pd.DataFrame([result])
    result_path = f"{args.out}.results.tsv"
    write_table(result_df, result_path)

    log_lines = [
        f"censoring_factor:{result['alpha']}",
        f"intercept: {result['intercept']} ({result['intercept_se']})",
        f"total heritability: {result['heritability']} ({result['heritability_se']})",
    ]
    log_path = f"{args.out}.log"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    return {"results": result_path, "log": log_path, "mode": result["mode"]}
