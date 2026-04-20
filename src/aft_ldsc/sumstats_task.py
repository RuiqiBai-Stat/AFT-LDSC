from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import gumbel_l, logistic, norm

from aft_ldsc.io_utils import (
    read_bim_like,
    read_fam_ids,
    read_table,
    resolve_bfile_chr_prefixes,
    write_alpha,
    write_table,
)
from aft_ldsc.qc import require_columns, validate_alpha

_EULER_GAMMA = 0.577215664901532


def _nelson_aalen_cumhaz_with_risk(resid: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resid = np.asarray(resid, dtype=float)
    event = np.asarray(event, dtype=int)
    u_grid = np.sort(np.unique(resid[event == 1]))
    if len(u_grid) == 0:
        return np.array([]), np.array([]), np.array([])
    a_np = np.zeros_like(u_grid, dtype=float)
    y_risk = np.zeros_like(u_grid, dtype=float)
    cum = 0.0
    for i, u in enumerate(u_grid):
        d_n = np.sum((resid == u) & (event == 1))
        y_u = np.sum(resid >= u)
        if y_u <= 0:
            continue
        cum += d_n / y_u
        a_np[i] = cum
        y_risk[i] = y_u
    return u_grid, a_np, y_risk


def _weighted_cvm(
    resid: np.ndarray,
    event: np.ndarray,
    cdf_fn: Callable[[np.ndarray], np.ndarray],
    min_risk: int = 5000,
    risk_power: float = 2.0,
) -> tuple[float, float]:
    u_grid, a_np, y_risk = _nelson_aalen_cumhaz_with_risk(resid, event)
    if len(u_grid) == 0:
        return np.nan, np.inf
    s_res = np.clip(1.0 - cdf_fn(u_grid), 1e-12, 1.0)
    a_p = -np.log(s_res)
    w = np.sqrt(len(resid)) * (a_p - a_np)
    q_stat = float(np.max(np.abs(w)))
    mask = y_risk >= min_risk
    if not np.any(mask):
        mask = np.ones_like(y_risk, dtype=bool)
    delta = a_p[mask] - a_np[mask]
    weights = (y_risk[mask] / len(resid)) ** risk_power
    wcvm = float(np.sum(weights * (delta**2)))
    return q_stat, wcvm


@dataclass
class _ModelResult:
    name: str
    fitter: object
    location_coef: np.ndarray
    q_stat: float
    wcvm: float


def _fit_and_score_models(
    t: np.ndarray,
    e: np.ndarray,
    cov_df: pd.DataFrame,
    min_risk: int,
    risk_power: float,
) -> list[_ModelResult]:
    try:
        from lifelines import LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter
    except ImportError as e:
        raise ImportError(
            "sumstats mode requires lifelines. Install dependencies with `pip install -r requirements.txt`."
        ) from e

    fit_df = pd.DataFrame({"T": t, "E": e, **{c: cov_df[c].values for c in cov_df.columns}})
    design = np.column_stack([np.ones(len(fit_df)), cov_df.values])
    coef_order = ["Intercept", *cov_df.columns.tolist()]
    results: list[_ModelResult] = []

    model_defs = [
        ("lognormal", LogNormalAFTFitter(), "mu_", lambda m: np.exp(m.params_.loc["sigma_"]["Intercept"]), lambda x, s: norm.cdf(x, loc=0, scale=s)),
        ("loglogistic", LogLogisticAFTFitter(), "alpha_", lambda m: np.exp(-m.params_.loc["beta_"]["Intercept"]), lambda x, s: logistic.cdf(x, loc=0, scale=s)),
        ("weibull", WeibullAFTFitter(), "lambda_", lambda m: np.exp(-m.params_.loc["rho_"]["Intercept"]), lambda x, s: gumbel_l.cdf(x, loc=0, scale=s)),
    ]

    for name, fitter, loc_row, scale_fn, cdf_base in model_defs:
        try:
            fitter.fit(fit_df, duration_col="T", event_col="E")
            coef_series = fitter.params_.loc[loc_row].reindex(coef_order)
            coef = coef_series.values.astype(float)
            resid = np.log(t) - design @ coef
            scale = float(scale_fn(fitter))
            q, wcvm = _weighted_cvm(resid, e, cdf_fn=lambda x: cdf_base(x, scale), min_risk=min_risk, risk_power=risk_power)
            results.append(_ModelResult(name=name, fitter=fitter, location_coef=coef, q_stat=q, wcvm=wcvm))
        except Exception:
            continue

    if not results:
        raise RuntimeError("All AFT model fits failed. Check input columns and event distribution.")
    return results


def _compute_raw_logy_alpha(
    name: str,
    fitter: object,
    coef: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    design: np.ndarray,
    censor_time: np.ndarray | None,
) -> tuple[np.ndarray, float]:
    try:
        from lifelines import LogLogisticFitter, LogNormalFitter, WeibullFitter
    except ImportError as ex:
        raise ImportError("sumstats mode requires lifelines.") from ex

    n = len(t)
    logy = np.zeros(n, dtype=float)
    xb = design @ coef
    mask = e == 1
    if name == "lognormal":
        res = np.log(t) - xb
        f = LogNormalFitter().fit(np.exp(res), event_observed=e)
        mu, sigma = float(f.mu_), float(f.sigma_)
        logy[mask] = (res[mask] - mu) / sigma
        if censor_time is None:
            return logy, float(np.clip(np.mean(e), 1e-6, 1.0 - 1e-6))
        logc = (np.log(censor_time) - xb - mu) / sigma
    elif name == "loglogistic":
        res = np.log(t) - xb
        f = LogLogisticFitter().fit(np.exp(res), event_observed=e)
        alpha_, beta_ = float(f.alpha_), float(f.beta_)
        std_logt = np.pi / (np.sqrt(3.0) * beta_)
        logy[mask] = (res[mask] - np.log(alpha_)) / std_logt
        if censor_time is None:
            return logy, float(np.clip(np.mean(e), 1e-6, 1.0 - 1e-6))
        logc = (np.log(censor_time) - xb - np.log(alpha_)) / std_logt
    elif name == "weibull":
        ratio = t / np.exp(xb)
        f = WeibullFitter().fit(ratio, event_observed=e)
        lam, rho = float(f.lambda_), float(f.rho_)
        mean_logt = np.log(lam) - (1.0 / rho) * _EULER_GAMMA
        std_logt = np.pi / (rho * np.sqrt(6.0))
        logy[mask] = (np.log(ratio[mask]) - mean_logt) / std_logt
        if censor_time is None:
            return logy, float(np.clip(np.mean(e), 1e-6, 1.0 - 1e-6))
        logc = (np.log(censor_time / np.exp(xb)) - mean_logt) / std_logt
    else:
        raise ValueError(f"Unknown model name: {name}")

    c = float(np.mean(norm.cdf(logc) - logc * norm.pdf(logc)))
    return logy, c


def _load_censor_time(path: str | None, n_expected: int) -> np.ndarray | None:
    if path is None:
        return None
    p = str(path)
    if p.lower().endswith(".npy"):
        arr = np.load(p)
    else:
        df = read_table(p)
        arr = df.iloc[:, -1].values if df.shape[1] == 1 else df["CENSOR_TIME"].values
    arr = np.asarray(arr, dtype=float).flatten()
    if arr.size != n_expected:
        raise ValueError(
            f"censor_time length ({arr.size}) does not match number of samples ({n_expected})."
        )
    return arr


def _parse_optional_columns(arg: str | None) -> list[str] | None:
    if arg is None or arg.strip() == "":
        return None
    return [c.strip() for c in arg.split(",") if c.strip()]


_WORKER: dict = {}


def _init_worker(
    bed_path: str,
    sample_loc: np.ndarray,
    design: np.ndarray,
    ctc_inv: np.ndarray,
    logy: np.ndarray,
    snp_ids: np.ndarray,
    bp_vals: np.ndarray,
    n_snps: int,
    blocksize: int,
    min_n_per_snp: int,
) -> None:
    from pysnptools.snpreader import Bed
    _WORKER["bed"] = Bed(bed_path, count_A1=False)
    _WORKER["sample_loc"] = sample_loc
    _WORKER["design"] = design
    _WORKER["ctc_inv"] = ctc_inv
    _WORKER["logy"] = logy
    _WORKER["snp_ids"] = snp_ids
    _WORKER["bp_vals"] = bp_vals
    _WORKER["n_snps"] = n_snps
    _WORKER["blocksize"] = blocksize
    _WORKER["min_n_per_snp"] = min_n_per_snp


def _compute_block(b: int) -> tuple[list[str], list[int], list[float], list[int]]:
    bed = _WORKER["bed"]
    sample_loc = _WORKER["sample_loc"]
    design = _WORKER["design"]
    ctc_inv = _WORKER["ctc_inv"]
    logy = _WORKER["logy"]
    snp_ids = _WORKER["snp_ids"]
    bp_vals = _WORKER["bp_vals"]
    n_snps = _WORKER["n_snps"]
    blocksize = _WORKER["blocksize"]
    min_n_per_snp = _WORKER["min_n_per_snp"]

    start = b * blocksize
    end = min(start + blocksize, n_snps)
    geno = bed[sample_loc, start:end].read().val
    col_mean = np.nanmean(geno, axis=0)
    col_std = np.nanstd(geno, axis=0)
    non_missing = (~np.isnan(geno)).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        geno_std = (geno - col_mean) / col_std

    proj = np.copy(geno_std)
    for j in range(geno_std.shape[1]):
        m = ~np.isnan(geno_std[:, j])
        if not np.any(m):
            continue
        x_m = geno_std[m, j]
        z_m = design[m]
        coef = ctc_inv @ (z_m.T @ x_m)
        proj[m, j] = x_m - z_m @ coef
    proj = np.nan_to_num(proj, nan=0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        x_scaled = proj / np.sqrt(non_missing)
    z_vals = logy @ x_scaled

    snps_out: list[str] = []
    n_out: list[int] = []
    z_out: list[float] = []
    bp_out: list[int] = []
    for j in range(end - start):
        nj = int(non_missing[j])
        if nj < min_n_per_snp or not np.isfinite(col_std[j]) or col_std[j] <= 0:
            continue
        snps_out.append(str(snp_ids[start + j]))
        n_out.append(nj)
        z_out.append(float(z_vals[j]))
        bp_out.append(int(bp_vals[start + j]))
    return snps_out, n_out, z_out, bp_out


def _stream_bed_sumstats(
    bfile: str,
    iid_keep: np.ndarray,
    logy: np.ndarray,
    design: np.ndarray,
    blocksize: int,
    min_n_per_snp: int,
    nproc: int = 1,
) -> pd.DataFrame:
    try:
        from pysnptools.snpreader import Bed
    except ImportError as ex:
        raise ImportError("bed input requires pysnptools.") from ex

    ctc_inv = np.linalg.pinv(design.T @ design)
    iid_keep = np.asarray(iid_keep).astype(str)

    frames: list[pd.DataFrame] = []
    for chrom, prefix in resolve_bfile_chr_prefixes(bfile):
        fam_ids = read_fam_ids(prefix + ".fam").astype(str)
        idx_map = {v: i for i, v in enumerate(fam_ids)}
        missing = [i for i in iid_keep if i not in idx_map]
        if missing:
            raise ValueError(
                f"{len(missing)} phenotype IIDs not found in {prefix}.fam (e.g. {missing[:3]})."
            )
        sample_loc = np.array([idx_map[i] for i in iid_keep], dtype=int)

        bim = read_bim_like(prefix + ".bim")
        snp_ids = bim["SNP"].values
        bp_vals = bim["BP"].values.astype(np.int64)
        n_snps = Bed(prefix + ".bed", count_A1=False).sid_count
        n_blocks = (n_snps + blocksize - 1) // blocksize

        init_args = (
            prefix + ".bed", sample_loc, design, ctc_inv, logy,
            snp_ids, bp_vals, n_snps, blocksize, min_n_per_snp,
        )

        snps_out: list[str] = []
        n_out: list[int] = []
        z_out: list[float] = []
        bp_out: list[int] = []

        if nproc <= 1:
            _init_worker(*init_args)
            block_iter = (_compute_block(b) for b in range(n_blocks))
        else:
            pool = ProcessPoolExecutor(max_workers=nproc, initializer=_init_worker, initargs=init_args)
            block_iter = pool.map(_compute_block, range(n_blocks))

        for s, n, z, bp in block_iter:
            snps_out.extend(s)
            n_out.extend(n)
            z_out.extend(z)
            bp_out.extend(bp)

        if nproc > 1:
            pool.shutdown()

        frame = pd.DataFrame(
            {"CHR": chrom, "SNP": snps_out, "BP": bp_out, "N": n_out, "Z": z_out}
        )
        frame = frame.merge(bim[["SNP", "A1", "A2"]], on="SNP", how="left")
        frame = frame.sort_values("BP").reset_index(drop=True)
        frames.append(frame)

    out = pd.concat(frames, axis=0, ignore_index=True)
    return out[["SNP", "A1", "A2", "Z", "N"]]


def _run_sumstats_bed(args, base: pd.DataFrame, covar_cols: list[str]) -> dict:
    merged = base.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[args.time_col, args.event_col]
    )
    t = pd.to_numeric(merged[args.time_col], errors="coerce").values
    e = pd.to_numeric(merged[args.event_col], errors="coerce").values.astype(float)
    mask = np.isfinite(t) & np.isfinite(e)
    merged = merged.loc[mask].copy()
    t = t[mask]
    e = e[mask]
    t = np.clip(t, 1e-6, None)
    e = (e > 0).astype(int)
    if e.sum() == 0:
        raise ValueError("No observed events after filtering.")

    bfile_paths = resolve_bfile_chr_prefixes(args.bfile)
    fam_id_sets = [set(read_fam_ids(p + ".fam").astype(str)) for _, p in bfile_paths]
    fam_ids_common = fam_id_sets[0]
    for s in fam_id_sets[1:]:
        fam_ids_common = fam_ids_common & s
    fam_ids_all = np.array(sorted(fam_ids_common))

    merged[args.iid_col] = merged[args.iid_col].astype(str)
    merged = merged[merged[args.iid_col].isin(set(fam_ids_all.astype(str)))].copy()
    merged = merged.sort_values(args.iid_col).reset_index(drop=True)
    if len(merged) == 0:
        raise ValueError("No overlap between phenotype IIDs and .fam IIDs.")

    t = pd.to_numeric(merged[args.time_col], errors="coerce").values
    e = (pd.to_numeric(merged[args.event_col], errors="coerce").values > 0).astype(int)
    t = np.clip(t, 1e-6, None)

    covar_numeric = (
        merged[covar_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if covar_cols
        else pd.DataFrame(index=merged.index)
    )
    design = np.column_stack([np.ones(len(merged)), covar_numeric.values]) if covar_cols else np.ones((len(merged), 1))

    model_results = _fit_and_score_models(
        t=t, e=e, cov_df=covar_numeric, min_risk=args.gof_min_riskset, risk_power=args.gof_risk_power
    )
    model_results = sorted(model_results, key=lambda r: r.wcvm)
    if args.aft_dist != "auto":
        selected = [r for r in model_results if r.name == args.aft_dist]
        if not selected:
            raise ValueError(f"Requested aft_dist={args.aft_dist}, but model fit failed.")
        best = selected[0]
    else:
        best = model_results[0]

    censor_time = _load_censor_time(args.censor_time, n_expected=len(merged))
    logy, alpha_raw = _compute_raw_logy_alpha(
        best.name, best.fitter, best.location_coef, t, e, design, censor_time
    )
    if args.alpha_value is not None:
        alpha = validate_alpha(args.alpha_value)
    else:
        alpha = validate_alpha(alpha_raw)

    nproc = getattr(args, "nproc", None)
    if nproc is None:
        nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    nproc = max(1, int(nproc))

    out = _stream_bed_sumstats(
        bfile=args.bfile,
        iid_keep=merged[args.iid_col].values,
        logy=logy,
        design=design,
        blocksize=args.blocksize,
        min_n_per_snp=args.min_n_per_snp,
        nproc=nproc,
    )
    return _write_sumstats_outputs(args, out, alpha, model_results, best, merged, e)


def _write_sumstats_outputs(
    args,
    out: pd.DataFrame,
    alpha: float,
    model_results: list[_ModelResult],
    best: _ModelResult,
    merged: pd.DataFrame,
    e: np.ndarray,
) -> dict:
    sumstats_path = f"{args.out}.sumstats.gz"
    write_table(out, sumstats_path)
    alpha_path = f"{args.out}.alpha"
    write_alpha(alpha_path, alpha)

    gof_df = pd.DataFrame(
        {
            "model": [m.name for m in model_results],
            "Q": [m.q_stat for m in model_results],
            "wCvM": [m.wcvm for m in model_results],
            "selected": [m.name == best.name for m in model_results],
        }
    )
    gof_path = f"{args.out}.gof.tsv"
    write_table(gof_df, gof_path)

    log_df = pd.DataFrame(
        [
            {
                "n_samples": len(merged),
                "n_events": int(e.sum()),
                "event_rate": float(np.mean(e)),
                "alpha": alpha,
                "best_model": best.name,
                "n_snps": len(out),
            }
        ]
    )
    log_path = f"{args.out}.log.tsv"
    write_table(log_df, log_path)
    return {"sumstats": sumstats_path, "alpha": alpha_path, "gof": gof_path, "log": log_path}


def run_sumstats(args) -> dict:
    pheno = read_table(args.input)
    require_columns(pheno, [args.iid_col, args.time_col], "phenotype input")

    if args.event is not None:
        event_df = read_table(args.event)
        require_columns(event_df, [args.iid_col, args.event_col], "event input")
        if args.event_col == args.time_col:
            event_col_internal = f"__event__{args.event_col}"
            ev_sub = event_df[[args.iid_col, args.event_col]].rename(
                columns={args.event_col: event_col_internal}
            )
            base = pheno[[args.iid_col, args.time_col]].merge(
                ev_sub, on=args.iid_col, how="inner"
            )
            args.event_col = event_col_internal
        else:
            base = pheno[[args.iid_col, args.time_col]].merge(
                event_df[[args.iid_col, args.event_col]], on=args.iid_col, how="inner"
            )
    else:
        require_columns(pheno, [args.event_col], "phenotype input (event column)")
        base = pheno[[args.iid_col, args.time_col, args.event_col]].copy()

    if args.covar is not None:
        covar_df = read_table(args.covar)
        require_columns(covar_df, [args.iid_col], "covariate input")
        covar_cols = _parse_optional_columns(args.covar_cols)
        if covar_cols is None:
            covar_cols = [c for c in covar_df.columns if c != args.iid_col]
        covar_df = covar_df[[args.iid_col, *covar_cols]]
        base = base.merge(covar_df, on=args.iid_col, how="inner")
    else:
        covar_cols = []

    return _run_sumstats_bed(args, base, covar_cols)
