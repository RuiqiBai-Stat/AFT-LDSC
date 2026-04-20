from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aft_ldsc",
        description="Minimal AFT-LDSC CLI (single-entry, LDSC-like IO).",
    )
    parser.add_argument("--mode", required=True, choices=["sumstats", "heritability", "enrichment", "gcor"])
    parser.add_argument("--out", required=True, help="Output prefix.")

    parser.add_argument("--input", help="Phenotype input for sumstats mode.")
    parser.add_argument("--sumstats", help="For heritability/enrichment: one .sumstats.gz. For gcor: two files comma-separated.")
    parser.add_argument("--alpha", help="For heritability/enrichment: one .alpha. For gcor: two files comma-separated.")

    parser.add_argument(
        "--ref-ld-chr",
        help="LDSC-style chromosome-split LD prefix (supports @ replacement). "
        "For celltype enrichment, pass two prefixes comma-separated (baseline,celltype_group).",
    )
    parser.add_argument("--w-ld-chr", help="LDSC-style chromosome-split regression weight LD prefix.")
    parser.add_argument("--frqfile-chr", help="LDSC-style chromosome-split frequency prefix.")
    parser.add_argument("--ld-col", default="L2", help="Preferred LD score column (default: L2).")
    parser.add_argument("--ld-cols", help="Comma-separated LD columns for enrichment; default uses all LD columns.")
    parser.add_argument("--weight-col", help="Optional LD weight column.")
    parser.add_argument("--annot", help="Optional annotation file for enrichment.")
    parser.add_argument("--annot-cols", help="Comma-separated annotation columns.")

    parser.add_argument("--n", type=float, help="Override sample size N for single-trait modes.")
    parser.add_argument("--n1", type=float, help="Override trait1 sample size for gcor.")
    parser.add_argument("--n2", type=float, help="Override trait2 sample size for gcor.")
    parser.add_argument("--m", type=int, help="Override number of SNPs M in formulas.")
    parser.add_argument("--n-blocks", type=int, default=200, help="Block jackknife blocks (default: 200).")
    parser.add_argument("--save-delete", action="store_true", help="Save jackknife delete values.")

    parser.add_argument("--alpha-value", type=float, help="Optional numeric alpha override for sumstats generation.")
    parser.add_argument("--overlap-annot", action="store_true", help="Reserved LDSC-compatible flag.")
    parser.add_argument("--print-coefficients", action="store_true", help="Reserved LDSC-compatible flag.")

    parser.add_argument("--event", help="Event file for sumstats mode.")
    parser.add_argument("--covar", help="Covariate file for sumstats mode.")
    parser.add_argument(
        "--bfile",
        help="PLINK bfile template for sumstats mode. Must contain a literal "
        "'@' placeholder; the tool replaces '@' with each chromosome 1..22 and "
        "streams all 22 chromosomes in one pass, writing a single combined "
        ".sumstats.gz.",
    )
    parser.add_argument(
        "--censor-time",
        help="Censoring-time file aligned with final sample order (.npy) or a single-column table.",
    )
    parser.add_argument("--blocksize", type=int, default=100, help="SNP block size for streaming .bed input.")
    parser.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="Number of worker processes for per-block parallelism. "
        "Defaults to SLURM_CPUS_PER_TASK env var if set, else 1.",
    )
    parser.add_argument("--iid-col", default="IID", help="IID column name (default: IID).")
    parser.add_argument("--time-col", default="TIME", help="Time column name (default: TIME).")
    parser.add_argument("--event-col", default="EVENT", help="Event indicator column name (default: EVENT).")
    parser.add_argument("--covar-cols", help="Comma-separated covariate columns to use.")
    parser.add_argument("--aft-dist", default="auto", choices=["auto", "lognormal", "loglogistic", "weibull"])
    parser.add_argument("--gof-min-riskset", type=int, default=5000)
    parser.add_argument("--gof-risk-power", type=float, default=2.0)
    parser.add_argument("--min-n-per-snp", type=int, default=50)

    return parser


def _validate_args(args) -> None:
    if args.mode == "sumstats":
        if args.input is None or args.bfile is None:
            raise ValueError("sumstats mode requires --input and --bfile.")
    elif args.mode in {"heritability", "enrichment"}:
        if args.sumstats is None or args.alpha is None or args.ref_ld_chr is None:
            raise ValueError(f"{args.mode} mode requires --sumstats --alpha --ref-ld-chr.")
    elif args.mode == "gcor":
        if args.sumstats is None or args.alpha is None or args.ref_ld_chr is None:
            raise ValueError("gcor mode requires --sumstats --alpha --ref-ld-chr.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        _validate_args(args)

        if args.mode == "sumstats":
            from aft_ldsc.sumstats_task import run_sumstats

            outputs = run_sumstats(args)
        elif args.mode == "heritability":
            from aft_ldsc.heritability_task import run_heritability

            outputs = run_heritability(args)
        elif args.mode == "enrichment":
            from aft_ldsc.enrichment_task import run_enrichment

            outputs = run_enrichment(args)
        elif args.mode == "gcor":
            from aft_ldsc.gcor_task import run_gcor

            outputs = run_gcor(args)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
    except Exception as e:
        raise SystemExit(f"Error: {e}") from e

    for k, v in outputs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
