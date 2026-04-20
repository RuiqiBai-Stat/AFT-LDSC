from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import t

from aft_ldsc.io_utils import (
    _resolve_with_compression,
    get_present_chrs,
    read_alpha,
    read_ldsc_table,
    split_csv_arg,
    sub_chr,
    write_table,
)
from aft_ldsc.qc import require_columns, validate_alpha


def _read_chr_table(prefix: str, chrom: int, suffix: str, sep: str) -> pd.DataFrame:
    base = sub_chr(prefix, chrom) + suffix
    path, comp = _resolve_with_compression(base)
    return pd.read_csv(path, sep=sep, compression=comp)


def _fast_block_jackknife(y: np.ndarray, x: np.ndarray, w: np.ndarray, n_blocks: int) -> tuple[np.ndarray, np.ndarray]:
    m = x.shape[0]
    k = x.shape[1]
    boundaries = np.floor(np.linspace(0, m, n_blocks + 1)).astype(int)
    xtx_block = np.zeros((n_blocks, k, k))
    xty_block = np.zeros((n_blocks, k))

    for i in range(n_blocks):
        s, e = boundaries[i], boundaries[i + 1]
        xb = x[s:e]
        yb = y[s:e]
        wb = w[s:e]
        xtx_block[i] = (xb.T * wb) @ xb
        xty_block[i] = (xb.T * wb) @ yb

    xtx_total = xtx_block.sum(axis=0)
    xty_total = xty_block.sum(axis=0)
    try:
        params_full = np.linalg.solve(xtx_total, xty_total)
    except np.linalg.LinAlgError:
        params_full = np.linalg.pinv(xtx_total) @ xty_total

    params_delete = np.zeros((n_blocks, k))
    for i in range(n_blocks):
        xtx_del = xtx_total - xtx_block[i]
        xty_del = xty_total - xty_block[i]
        try:
            params_delete[i] = np.linalg.solve(xtx_del, xty_del)
        except np.linalg.LinAlgError:
            params_delete[i] = np.linalg.pinv(xtx_del) @ xty_del

    pseudovalues = n_blocks * params_full - (n_blocks - 1) * params_delete
    cov = np.cov(pseudovalues.T, ddof=1) / n_blocks
    return params_full, params_delete, cov


def _celltype_label_from_prefix(prefix: str, fallback: str) -> str:
    m = re.search(r"cell_type_group\.(\d+)\.", prefix)
    if m:
        return f"Cell_type_group_{m.group(1)}"
    return fallback


def _celltype_annot_prefix_for_compat(prefix: str) -> str:
    """
    Keep compatibility with raw_script/celltype.py, which uses:
    cell_type_group.1.{chr}.annot.gz for all celltype runs.
    """
    if "cell_type_group." in prefix:
        return re.sub(r"cell_type_group\.\d+\.", "cell_type_group.1.", prefix)
    return prefix


def run_enrichment(args) -> dict:
    sumstats = read_ldsc_table(args.sumstats)
    require_columns(sumstats, ["SNP", "N", "Z", "A1", "A2"], "sumstats")
    alpha = validate_alpha(read_alpha(args.alpha))

    if args.ref_ld_chr is None:
        raise ValueError("enrichment requires --ref-ld-chr.")
    if args.w_ld_chr is None:
        raise ValueError("enrichment requires --w-ld-chr.")
    if args.frqfile_chr is None:
        raise ValueError("enrichment requires --frqfile-chr.")

    ref_prefixes = split_csv_arg(args.ref_ld_chr)
    baseline_prefix = ref_prefixes[0]
    extra_prefixes = ref_prefixes[1:]

    chrs = get_present_chrs(baseline_prefix, max_chr=22)
    if not chrs:
        raise ValueError(f"No chromosome files found for --ref-ld-chr {baseline_prefix}")

    ldscore_list = []
    annot_list = []
    weight_list = []
    z_list = []
    n_list = []

    annot_name = None

    for chrom in chrs:
        baseline_ld = _read_chr_table(baseline_prefix, chrom, ".l2.ldscore", sep="\t")
        require_columns(baseline_ld, ["SNP"], f"ref ld chr{chrom}")
        extra_ld_list = [_read_chr_table(p, chrom, ".l2.ldscore", sep="\t") for p in extra_prefixes]

        hm3_ld = _read_chr_table(args.w_ld_chr, chrom, ".l2.ldscore", sep="\t")
        require_columns(hm3_ld, ["SNP", "L2"], f"w ld chr{chrom}")

        frq = _read_chr_table(args.frqfile_chr, chrom, ".frq", sep=r"\s+")
        if "MAF" in frq.columns:
            frq_mask = frq["MAF"].values >= 0.05
        elif "FRQ" in frq.columns:
            frq_mask = (frq["FRQ"].values >= 0.05) & (frq["FRQ"].values <= 0.95)
        else:
            raise ValueError(f"frq chr{chrom} missing MAF/FRQ column.")

        annot_chr = _read_chr_table(baseline_prefix, chrom, ".annot", sep="\t")
        if annot_chr.shape[1] < 5:
            raise ValueError(f"annot chr{chrom} has too few columns.")
        ann_blocks = [annot_chr.iloc[np.where(frq_mask)[0], 4:].values]
        if annot_name is None:
            annot_name = list(annot_chr.columns[4:])

        # Multi-prefix mode for celltype: append each extra annot column (compat with raw script).
        for p in extra_prefixes:
            ann_prefix = _celltype_annot_prefix_for_compat(p)
            extra_annot = _read_chr_table(ann_prefix, chrom, ".annot", sep="\t")
            if extra_annot.shape[1] < 5:
                raise ValueError(f"extra annot chr{chrom} has too few columns for prefix {ann_prefix}")
            ann_blocks.append(extra_annot.iloc[np.where(frq_mask)[0], 4].values.reshape(-1, 1))
            if chrom == chrs[0]:
                annot_name.append(_celltype_label_from_prefix(p, str(extra_annot.columns[4])))

        # Follow baseline/celltype scripts exactly: filter annot by frq index mask, not by overlap SNPs.
        annot_list.append(np.hstack(ann_blocks))

        overlap_variant = np.intersect1d(hm3_ld["SNP"].values, np.intersect1d(baseline_ld["SNP"].values, sumstats["SNP"].values))
        base_mask = baseline_ld["SNP"].isin(overlap_variant)
        ld_blocks = [baseline_ld.loc[base_mask].iloc[:, 3:].values]
        for extra_ld in extra_ld_list:
            if "L2" in extra_ld.columns:
                ld_blocks.append(extra_ld.loc[base_mask, "L2"].values.reshape(-1, 1))
            else:
                extra_cols = [c for c in extra_ld.columns if c not in {"CHR", "SNP", "BP", "CM", "MAF"}]
                if not extra_cols:
                    raise ValueError("extra ldscore file has no usable LD column.")
                ld_blocks.append(extra_ld.loc[base_mask, extra_cols[0]].values.reshape(-1, 1))

        ldscore_list.append(np.hstack(ld_blocks))
        weight_list.append(hm3_ld[hm3_ld["SNP"].isin(overlap_variant)]["L2"].values)
        z_list.append(sumstats[sumstats["SNP"].isin(overlap_variant)]["Z"].values)
        n_list.append(sumstats[sumstats["SNP"].isin(overlap_variant)]["N"].values)

    ldscore = np.vstack(ldscore_list)
    annot = np.vstack(annot_list)
    weight = np.concatenate(weight_list).astype(float)
    weight[weight < 1] = 1
    z = np.concatenate(z_list).astype(float)
    y = z**2
    n_vec = np.concatenate(n_list).astype(float).reshape(-1, 1)

    if len(y) == 0:
        raise ValueError("No SNPs left after overlap with LD panels.")

    nbar = float(np.mean(n_vec).astype(int))
    if nbar <= 0:
        raise ValueError("Invalid average sample size (Nbar <= 0).")

    ldscore = ldscore * n_vec / nbar

    x = np.column_stack([np.ones(len(y)), ldscore * (alpha**2)])
    m = len(y)
    k = x.shape[1]

    tau_hat = (np.mean(y) - alpha) / (nbar * np.mean(np.sum(ldscore, axis=1)) * (alpha**2))
    w = 1.0 / weight / (alpha + nbar * tau_hat * np.sum(ldscore, axis=1) * (alpha**2)) ** 2

    n_blocks = int(args.n_blocks)
    if n_blocks <= 1:
        raise ValueError("--n-blocks must be > 1.")

    params_full_raw, params_delete_raw, cov_raw = _fast_block_jackknife(y=y, x=x, w=w, n_blocks=n_blocks)
    params_full = params_full_raw / nbar
    params_delete = params_delete_raw / nbar
    cov = cov_raw / (nbar**2)
    params_se = np.sqrt(np.diag(cov))

    annot_sum = np.sum(annot, axis=0)
    intercept = float(params_full[0] * nbar / alpha)
    total_heri = float(annot_sum @ params_full[1:])
    intercept_se = float(params_se[0] * nbar / alpha)
    total_heri_se = float(np.sqrt(annot_sum @ cov[1:, 1:] @ annot_sum.T))

    total_heri_snp_size = annot.shape[0]
    annot_heri = annot_sum * params_full[1:]
    annot_heri_prop = annot_heri / total_heri
    total_heri_jk = params_delete[:, 1:] @ annot_sum
    annot_heri_jk = params_delete[:, 1:] * annot_sum
    annot_heri_prop_jk = annot_heri_jk / total_heri_jk.reshape(-1, 1)
    pseudovalues_heri_prop_jk = n_blocks * annot_heri_prop - (n_blocks - 1) * annot_heri_prop_jk
    cov_matrix_heri_prop = np.cov(pseudovalues_heri_prop_jk.T, ddof=1) / n_blocks

    annot_k_heri_prop = []
    annot_k_heri_prop_se = []
    annot_k_snp_prop = []
    annot_k_enrichment = []
    annot_k_enrichment_se = []
    annot_k_enrichment_p = []

    for idx in range(1, k):
        ann_col = annot[:, idx - 1]
        if np.all(np.isin(ann_col, [0, 1])):
            ann_sum_k = np.sum(annot[ann_col == 1], axis=0)
            ann_prop = ann_sum_k / annot_sum
            prop_h2 = ann_prop @ annot_heri_prop
            prop_snps = np.sum(ann_col == 1) / total_heri_snp_size
            enrich = prop_h2 / prop_snps
            annot_k_heri_prop.append(prop_h2)
            annot_k_snp_prop.append(prop_snps)
            annot_k_enrichment.append(enrich)

            if np.all(ann_col == 1):
                annot_k_heri_prop_se.append("NA")
                annot_k_enrichment_se.append("NA")
                annot_k_enrichment_p.append("NA")
            else:
                prop_h2_se = np.sqrt(ann_prop @ cov_matrix_heri_prop @ ann_prop.T)
                enrich_se = prop_h2_se / prop_snps
                diff_vec = ann_sum_k / np.sum(ann_col == 1) - (annot_sum - ann_sum_k) / (
                    total_heri_snp_size - np.sum(ann_col == 1)
                )
                diff = diff_vec @ params_full[1:]
                diff_se = np.sqrt(diff_vec @ cov[1:, 1:] @ diff_vec.T)
                enrich_p = 2 * t.sf(abs(diff / diff_se), df=n_blocks)
                annot_k_heri_prop_se.append(prop_h2_se)
                annot_k_enrichment_se.append(enrich_se)
                annot_k_enrichment_p.append(enrich_p)
        else:
            annot_k_heri_prop.append("NA")
            annot_k_heri_prop_se.append("NA")
            annot_k_snp_prop.append("NA")
            annot_k_enrichment.append("NA")
            annot_k_enrichment_se.append("NA")
            annot_k_enrichment_p.append("NA")

    results = pd.DataFrame(
        {
            "Category": annot_name,
            "Prop._SNPs": annot_k_snp_prop,
            "Prop._h2": annot_k_heri_prop,
            "Prop._h2_std_error": annot_k_heri_prop_se,
            "Enrichment": annot_k_enrichment,
            "Enrichment_std_error": annot_k_enrichment_se,
            "Enrichment_p": annot_k_enrichment_p,
            "Coefficient": params_full[1:],
            "Coefficient_std_error": params_se[1:],
            "Coefficient_z-score": params_full[1:] / params_se[1:],
        }
    )

    result_path = f"{args.out}.results.tsv"
    write_table(results, result_path)
    result_csv_path = f"{args.out}_results.csv"
    results.to_csv(result_csv_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "censoring_factor": alpha,
                "intercept": intercept,
                "intercept_se": intercept_se,
                "total_heritability": total_heri,
                "total_heritability_se": total_heri_se,
                "n_snps_regression": int(m),
                "n_snps_annotation": int(total_heri_snp_size),
                "nbar": nbar,
            }
        ]
    )
    summary_path = f"{args.out}.summary.tsv"
    write_table(summary, summary_path)

    log_lines = [
        f"censoring_factor:{alpha}",
        f"intercept: {intercept} ({intercept_se})",
        f"total heritability: {total_heri} ({total_heri_se})",
    ]
    log_path = f"{args.out}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    if args.save_delete:
        delete_df = pd.DataFrame(params_delete, columns=["intercept", *list(annot_name)])
        write_table(delete_df, f"{args.out}.delete.tsv")

    return {"results": result_path, "results_csv": result_csv_path, "summary": summary_path, "log": log_path}
