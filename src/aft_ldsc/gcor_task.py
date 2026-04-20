from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aft_ldsc.heritability_task import _compute_ld_mode, _jackknife_scaled, _read_chr_table
from aft_ldsc.io_utils import get_present_chrs, read_alpha, read_ldsc_table, split_csv_arg, write_table
from aft_ldsc.qc import drop_nonfinite, require_columns, validate_alpha


def _align_alleles(df: pd.DataFrame) -> pd.DataFrame:
    same = (df["A1_1"] == df["A1_2"]) & (df["A2_1"] == df["A2_2"])
    swap = (df["A1_1"] == df["A2_2"]) & (df["A2_1"] == df["A1_2"])
    keep = same | swap
    out = df.loc[keep].copy()
    flip = swap.loc[keep].values
    out.loc[flip, "Z_2"] = -out.loc[flip, "Z_2"]
    return out


def _weighted_solve(y: np.ndarray, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xtx = (x.T * w) @ x
    xty = (x.T * w) @ y
    try:
        return np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx) @ xty


def _read_pair_overlap(
    sumstats1: pd.DataFrame,
    sumstats2: pd.DataFrame,
    ref_prefix: str,
    ld_col: str,
    frq_prefix: str | None,
) -> dict:
    chrs = get_present_chrs(ref_prefix, max_chr=22)
    if not chrs:
        raise ValueError(f"No chromosome files found for --ref-ld-chr {ref_prefix}")

    t1 = sumstats1[["SNP", "A1", "A2", "Z", "N"]]
    t2 = sumstats2[["SNP", "A1", "A2", "Z", "N"]]

    ld_list = []
    z1_list = []
    z2_list = []
    n1_list = []
    n2_list = []

    for chrom in chrs:
        ld_chr = _read_chr_table(ref_prefix, chrom, ".l2.ldscore")
        require_columns(ld_chr, ["SNP", ld_col], f"ref ld chr{chrom}")
        ld_chr = ld_chr[["SNP", ld_col]]

        merged = (
            ld_chr.merge(t1, on="SNP", how="inner")
            .merge(t2, on="SNP", how="inner", suffixes=("_1", "_2"))
        )
        merged = _align_alleles(merged)

        if frq_prefix is not None:
            frq = _read_chr_table(frq_prefix, chrom, ".frq", sep=r"\s+")
            require_columns(frq, ["SNP"], f"frq chr{chrom}")
            if "MAF" in frq.columns:
                maf = frq[["SNP", "MAF"]].rename(columns={"MAF": "__MAF__"})
            elif "FRQ" in frq.columns:
                maf = frq[["SNP", "FRQ"]].copy()
                maf["__MAF__"] = np.minimum(maf["FRQ"].values, 1.0 - maf["FRQ"].values)
                maf = maf[["SNP", "__MAF__"]]
            else:
                raise ValueError(f"frq chr{chrom} missing MAF/FRQ column.")
            merged = merged.merge(maf, on="SNP", how="inner")
            merged = merged[merged["__MAF__"] > 0.05]

        merged = drop_nonfinite(merged, ["Z_1", "Z_2", "N_1", "N_2", ld_col])
        if len(merged) == 0:
            continue

        ld_list.append(pd.to_numeric(merged[ld_col], errors="coerce").values.astype(float))
        z1_list.append(pd.to_numeric(merged["Z_1"], errors="coerce").values.astype(float))
        z2_list.append(pd.to_numeric(merged["Z_2"], errors="coerce").values.astype(float))
        n1_list.append(pd.to_numeric(merged["N_1"], errors="coerce").values.astype(float))
        n2_list.append(pd.to_numeric(merged["N_2"], errors="coerce").values.astype(float))

    if not ld_list:
        raise ValueError("No overlapping SNPs between two sumstats and LD panel.")

    ld = np.concatenate(ld_list).astype(float)
    z1 = np.concatenate(z1_list).astype(float)
    z2 = np.concatenate(z2_list).astype(float)
    n1 = np.concatenate(n1_list).astype(float)
    n2 = np.concatenate(n2_list).astype(float)

    return {"ld": ld, "z1": z1, "z2": z2, "n1": n1, "n2": n2}


def _estimate_gcov(
    z1: np.ndarray,
    z2: np.ndarray,
    ld_cross: np.ndarray,
    alpha1: float,
    alpha2: float,
    nbar1: float,
    nbar2: float,
    h2_1: float,
    h2_2: float,
    intercept_1: float,
    intercept_2: float,
    m_total: float,
    n_blocks: int,
) -> dict:
    n_cross = float(np.sqrt(nbar1 * nbar2))
    zprod = z1 * z2
    x = np.column_stack([np.ones(len(ld_cross)), ld_cross * alpha1 * alpha2])

    init_num = np.mean(zprod * m_total)
    init_den = np.mean(ld_cross * n_cross * alpha1 * alpha2)
    init_gcov = init_num / max(init_den, 1e-12)

    a = (nbar1 * h2_1 * (alpha1**2) / m_total) * ld_cross + intercept_1
    b = (nbar2 * h2_2 * (alpha2**2) / m_total) * ld_cross + intercept_2
    c = (n_cross * init_gcov * alpha1 * alpha2 / m_total) * ld_cross
    w = 1.0 / np.clip(a * b + c**2, 1e-12, None) / np.clip(ld_cross, 1e-12, None)

    beta = _weighted_solve(y=zprod, x=x, w=w)
    intercept_old = float(beta[0])
    gcov_old = float(beta[1] * m_total / n_cross)
    intercept = 10.0
    gcov = 10.0

    max_iter = 1000
    n_iter = 0
    while abs(gcov - gcov_old) > 1e-8 and n_iter < max_iter:
        intercept_old = float(beta[0])
        gcov_old = float(beta[1] * m_total / n_cross)
        c = (n_cross * gcov_old * alpha1 * alpha2 / m_total) * ld_cross + intercept_old
        w = 1.0 / np.clip(a * b + c**2, 1e-12, None) / np.clip(ld_cross, 1e-12, None)
        beta = _weighted_solve(y=zprod, x=x, w=w)
        intercept = float(beta[0])
        gcov = float(beta[1] * m_total / n_cross)
        n_iter += 1

    if n_iter >= max_iter:
        intercept = intercept_old
        gcov = gcov_old

    _, delete, cov = _jackknife_scaled(y=zprod, x=x, w=w, n_blocks=n_blocks, scale=1.0)
    gcov_delete = delete[:, 1] * m_total / n_cross
    gcov_se = float(np.sqrt(cov[1, 1]) * m_total / n_cross)
    intercept_se = float(np.sqrt(cov[0, 0]))

    return {
        "gcov": gcov,
        "gcov_se": gcov_se,
        "gcov_delete": gcov_delete,
        "intercept": intercept,
        "intercept_se": intercept_se,
        "n_cross": n_cross,
    }


def run_gcor(args) -> dict:
    sumstats_files = split_csv_arg(args.sumstats)
    alpha_files = split_csv_arg(args.alpha)
    ref_prefixes = split_csv_arg(args.ref_ld_chr)

    if len(sumstats_files) != 2:
        raise ValueError("gcor requires exactly two files in --sumstats (comma-separated).")
    if len(alpha_files) != 2:
        raise ValueError("gcor requires exactly two files in --alpha (comma-separated).")
    if len(ref_prefixes) != 1:
        raise ValueError("gcor currently expects exactly one --ref-ld-chr prefix.")

    ref_prefix = ref_prefixes[0]
    ld_col = args.ld_col if args.ld_col is not None else "L2"
    n_blocks = int(args.n_blocks)
    if n_blocks <= 1:
        raise ValueError("--n-blocks must be > 1.")

    sumstats1 = read_ldsc_table(sumstats_files[0])
    sumstats2 = read_ldsc_table(sumstats_files[1])
    require_columns(sumstats1, ["SNP", "N", "Z", "A1", "A2"], "sumstats1")
    require_columns(sumstats2, ["SNP", "N", "Z", "A1", "A2"], "sumstats2")

    alpha1 = validate_alpha(read_alpha(alpha_files[0]))
    alpha2 = validate_alpha(read_alpha(alpha_files[1]))

    heri1 = _compute_ld_mode(
        sumstats=sumstats1,
        alpha=alpha1,
        ref_prefix=ref_prefix,
        n_blocks=n_blocks,
        m_override=args.m,
        return_jackknife=True,
    )
    heri2 = _compute_ld_mode(
        sumstats=sumstats2,
        alpha=alpha2,
        ref_prefix=ref_prefix,
        n_blocks=n_blocks,
        m_override=args.m,
        return_jackknife=True,
    )

    overlap = _read_pair_overlap(
        sumstats1=sumstats1,
        sumstats2=sumstats2,
        ref_prefix=ref_prefix,
        ld_col=ld_col,
        frq_prefix=args.frqfile_chr,
    )

    base_ld = overlap["ld"].copy()
    base_ld[base_ld < 1.0] = 1.0
    z1 = overlap["z1"]
    z2 = overlap["z2"]
    n1_vec = overlap["n1"]
    n2_vec = overlap["n2"]

    nbar1 = float(args.n1 if args.n1 is not None else heri1["nbar"])
    nbar2 = float(args.n2 if args.n2 is not None else heri2["nbar"])
    if nbar1 <= 0 or nbar2 <= 0:
        raise ValueError("N1/N2 must be > 0.")

    m_total = float(args.m if args.m is not None else heri1["m_total"])
    ld_cross = base_ld * np.sqrt(n1_vec * n2_vec) / np.sqrt(nbar1 * nbar2)

    gcov_res = _estimate_gcov(
        z1=z1,
        z2=z2,
        ld_cross=ld_cross,
        alpha1=alpha1,
        alpha2=alpha2,
        nbar1=nbar1,
        nbar2=nbar2,
        h2_1=float(heri1["heritability"]),
        h2_2=float(heri2["heritability"]),
        intercept_1=float(heri1["intercept"]),
        intercept_2=float(heri2["intercept"]),
        m_total=m_total,
        n_blocks=n_blocks,
    )

    h2_jk_1 = np.clip(np.asarray(heri1["h2_jk_delete"], dtype=float), 0.001, 1.0)
    h2_jk_2 = np.clip(np.asarray(heri2["h2_jk_delete"], dtype=float), 0.001, 1.0)
    gcov_delete = np.asarray(gcov_res["gcov_delete"], dtype=float)

    h2_1 = float(heri1["heritability"])
    h2_2 = float(heri2["heritability"])
    rg = float(gcov_res["gcov"] / np.sqrt(max(h2_1 * h2_2, 1e-12)))

    rg_delete = gcov_delete / np.sqrt(h2_jk_1 * h2_jk_2)
    n_blocks = len(rg_delete)
    pseudo = (n_blocks * rg) - (n_blocks - 1) * rg_delete
    rg_se = float(np.sqrt(np.cov(pseudo.T, ddof=1) / n_blocks))

    result = pd.DataFrame(
        [
            {
                "sumstats1": sumstats_files[0],
                "sumstats2": sumstats_files[1],
                "alpha1": alpha1,
                "alpha2": alpha2,
                "n_overlap_snps": int(len(z1)),
                "nbar1": nbar1,
                "nbar2": nbar2,
                "m_total": m_total,
                "h2_1": h2_1,
                "h2_1_se": float(heri1["heritability_se"]),
                "h2_2": h2_2,
                "h2_2_se": float(heri2["heritability_se"]),
                "intercept_1": float(heri1["intercept"]),
                "intercept_1_se": float(heri1["intercept_se"]),
                "intercept_2": float(heri2["intercept"]),
                "intercept_2_se": float(heri2["intercept_se"]),
                "gcov": float(gcov_res["gcov"]),
                "gcov_se": float(gcov_res["gcov_se"]),
                "gcor_intercept": float(gcov_res["intercept"]),
                "gcor_intercept_se": float(gcov_res["intercept_se"]),
                "gcor": rg,
                "gcor_se": rg_se,
            }
        ]
    )

    result_path = f"{args.out}.results.tsv"
    write_table(result, result_path)

    log_lines = [
        f"censoring_factor_1:{alpha1}",
        f"censoring_factor_2:{alpha2}",
        f"intercept_1: {heri1['intercept']} ({heri1['intercept_se']})",
        f"intercept_2: {heri2['intercept']} ({heri2['intercept_se']})",
        f"heritability_1: {h2_1} ({heri1['heritability_se']})",
        f"heritability_2: {h2_2} ({heri2['heritability_se']})",
        f"gcor_intercept: {gcov_res['intercept']} ({gcov_res['intercept_se']})",
        f"gcor: {rg} ({rg_se})",
    ]
    log_path = f"{args.out}.log"
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    outputs = {"results": result_path, "log": log_path}
    if args.save_delete:
        delete_df = pd.DataFrame(
            {
                "gcov_delete": gcov_delete,
                "h2_1_delete": h2_jk_1,
                "h2_2_delete": h2_jk_2,
                "gcor_delete": rg_delete,
            }
        )
        delete_path = f"{args.out}.delete.tsv"
        write_table(delete_df, delete_path)
        outputs["delete"] = delete_path

    return outputs
