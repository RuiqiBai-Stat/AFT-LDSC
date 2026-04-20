"""
A runnable tutorial for AFT-LDSC using packaged example sumstats.

This script demonstrates:
1) heritability (LDSC panel)
2) heritability (baselineLD panel)
3) enrichment (baselineLD)
4) enrichment (baselineLD + cell type group 1)
5) genetic correlation (LDSC panel)

LD panel note:
The LD panel files are large and are not bundled in this repository.
Please download the needed LDSC resources to your local machine from:
https://console.cloud.google.com/storage/browser/broad-alkesgroup-public-requester-pays/LDSCORE?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def _require_chr_ld_prefix(prefix: str) -> None:
    p = Path(f"{prefix}1.l2.ldscore")
    gz = Path(f"{prefix}1.l2.ldscore.gz")
    bz2 = Path(f"{prefix}1.l2.ldscore.bz2")
    if not (p.exists() or gz.exists() or bz2.exists()):
        raise FileNotFoundError(
            f"Cannot find chr1 LD score file for prefix: {prefix}\n"
            "Expected one of: .l2.ldscore, .l2.ldscore.gz, .l2.ldscore.bz2"
        )


def _require_chr_frq_prefix(prefix: str) -> None:
    p = Path(f"{prefix}1.frq")
    gz = Path(f"{prefix}1.frq.gz")
    bz2 = Path(f"{prefix}1.frq.bz2")
    if not (p.exists() or gz.exists() or bz2.exists()):
        raise FileNotFoundError(
            f"Cannot find chr1 FRQ file for prefix: {prefix}\n"
            "Expected one of: .frq, .frq.gz, .frq.bz2"
        )


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples"
    out_dir = examples_dir / "result"
    out_dir.mkdir(parents=True, exist_ok=True)

    sumstats_250 = examples_dir / "250.2.sumstats.gz"
    alpha_250 = examples_dir / "250.2.alpha"
    sumstats_401 = examples_dir / "401.1.sumstats.gz"
    alpha_401 = examples_dir / "401.1.alpha"
    _require_file(sumstats_250)
    _require_file(alpha_250)
    _require_file(sumstats_401)
    _require_file(alpha_401)

    ldpanel_root = Path(
        os.environ.get("AFT_LDSC_LDPANEL", "/Users/ruiqi/Desktop/survivalGWAS/LDpanel")
    )

    # LDSC-style EUR LD
    ldsc_prefix = str(ldpanel_root / "eur_w_ld_chr") + "/"

    # baselineLD v2.2
    baseline_prefix = str(
        ldpanel_root / "1000G_Phase3_baselineLD_v2.2_ldscores" / "baselineLD."
    )
    # HM3 weights
    weight_prefix = str(
        ldpanel_root
        / "1000G_Phase3_weights_hm3_no_MHC"
        / "weights.hm3_noMHC."
    )
    # FRQ
    frq_prefix = str(ldpanel_root / "1000G_Phase3_frq" / "1000G.EUR.QC.")
    # Cell type group 1
    celltype1_prefix = str(
        ldpanel_root / "1000G_Phase3_cell_type_groups" / "cell_type_group.1."
    )

    _require_chr_ld_prefix(ldsc_prefix)
    _require_chr_ld_prefix(baseline_prefix)
    _require_chr_ld_prefix(weight_prefix)
    _require_chr_ld_prefix(celltype1_prefix)
    _require_chr_frq_prefix(frq_prefix)

    n_blocks = os.environ.get("AFT_LDSC_N_BLOCKS", "200")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")

    print(
        "LD panel files are large and must be downloaded locally first.\n"
        "If missing, get them from:\n"
        "https://console.cloud.google.com/storage/browser/broad-alkesgroup-public-requester-pays/LDSCORE?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))\n"
    )

    # 1) Heritability with LDSC panel
    _run(
        [
            sys.executable,
            "-m",
            "aft_ldsc.core",
            "--mode",
            "heritability",
            "--sumstats",
            str(sumstats_250),
            "--alpha",
            str(alpha_250),
            "--ref-ld-chr",
            ldsc_prefix,
            "--n-blocks",
            n_blocks,
            "--out",
            str(out_dir / "250.2_h2_ldsc"),
        ],
        env=env,
    )

    # 2) Heritability with baselineLD panel
    _run(
        [
            sys.executable,
            "-m",
            "aft_ldsc.core",
            "--mode",
            "heritability",
            "--sumstats",
            str(sumstats_250),
            "--alpha",
            str(alpha_250),
            "--ref-ld-chr",
            baseline_prefix,
            "--w-ld-chr",
            weight_prefix,
            "--frqfile-chr",
            frq_prefix,
            "--n-blocks",
            n_blocks,
            "--out",
            str(out_dir / "250.2_h2_baselineLD"),
        ],
        env=env,
    )

    # 3) Enrichment with baselineLD only
    _run(
        [
            sys.executable,
            "-m",
            "aft_ldsc.core",
            "--mode",
            "enrichment",
            "--sumstats",
            str(sumstats_250),
            "--alpha",
            str(alpha_250),
            "--ref-ld-chr",
            baseline_prefix,
            "--w-ld-chr",
            weight_prefix,
            "--frqfile-chr",
            frq_prefix,
            "--n-blocks",
            n_blocks,
            "--out",
            str(out_dir / "250.2_enrichment_baselineLD"),
        ],
        env=env,
    )

    # 4) Enrichment with baselineLD + cell type group 1
    _run(
        [
            sys.executable,
            "-m",
            "aft_ldsc.core",
            "--mode",
            "enrichment",
            "--sumstats",
            str(sumstats_250),
            "--alpha",
            str(alpha_250),
            "--ref-ld-chr",
            f"{baseline_prefix},{celltype1_prefix}",
            "--w-ld-chr",
            weight_prefix,
            "--frqfile-chr",
            frq_prefix,
            "--n-blocks",
            n_blocks,
            "--out",
            str(out_dir / "250.2_enrichment_baselineLD_celltype1"),
        ],
        env=env,
    )

    # 5) Genetic correlation with LDSC panel
    _run(
        [
            sys.executable,
            "-m",
            "aft_ldsc.core",
            "--mode",
            "gcor",
            "--sumstats",
            f"{sumstats_250},{sumstats_401}",
            "--alpha",
            f"{alpha_250},{alpha_401}",
            "--ref-ld-chr",
            ldsc_prefix,
            "--n-blocks",
            n_blocks,
            "--out",
            str(out_dir / "250.2_401.1_gcor_ldsc"),
        ],
        env=env,
    )

    print(f"Tutorial finished. Results are in: {out_dir}")


if __name__ == "__main__":
    main()
