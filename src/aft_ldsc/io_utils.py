from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def infer_sep(path: str | Path) -> str:
    name = str(path).lower()
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return ","
    return "\t"


def read_table(path: str | Path, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    sep = infer_sep(path)
    return pd.read_csv(path, sep=sep, usecols=usecols)


def read_ldsc_table(path: str | Path, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    """
    LDSC-style text parser: whitespace-separated with optional compression.
    """
    return pd.read_csv(path, sep=r"\s+", usecols=usecols)


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    ensure_parent_dir(path)
    sep = infer_sep(path)
    compression = "gzip" if str(path).lower().endswith(".gz") else None
    df.to_csv(path, sep=sep, index=False, compression=compression)


def write_alpha(path: str | Path, alpha: float) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{float(alpha):.12g}\n")


def read_alpha(path: str | Path) -> float:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError(f"Empty alpha file: {path}")
    token = content.split()[0]
    return float(token)


def read_bim_like(path: str | Path) -> pd.DataFrame:
    """
    Read PLINK .bim-like file.
    Output columns: CHR, SNP, BP, A1, A2
    """
    df = pd.read_csv(path, sep="\t|\\s+", engine="python", header=None)
    if df.shape[1] < 6:
        raise ValueError(f"BIM file {path} must have at least 6 columns.")
    out = pd.DataFrame(
        {
            "CHR": df.iloc[:, 0],
            "SNP": df.iloc[:, 1],
            "BP": df.iloc[:, 3],
            "A1": df.iloc[:, 4],
            "A2": df.iloc[:, 5],
        }
    )
    return out


def read_fam_ids(path: str | Path, id_col: int = 0) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] < 2:
        raise ValueError(f"FAM file {path} must have at least 2 columns.")
    return df.iloc[:, id_col].astype(str).values


def resolve_bfile_chr_prefixes(bfile: str) -> list[tuple[int, str]]:
    """
    Resolve a PLINK bfile argument into 22 per-chromosome prefixes.

    The argument must contain a literal '@' placeholder; '@' is replaced with
    each chromosome 1..22 in turn (LDSC-style). All 22 .bed files must exist
    on disk.

    Returns a list of (chrom, prefix) tuples in order 1..22, where prefix has
    no .bed suffix.
    """
    s = str(bfile)
    if s.endswith(".bed"):
        s = s[:-4]
    if "@" not in s:
        raise ValueError(
            f"--bfile must contain a literal '@' placeholder (e.g., /path/ukb_chr@_clean); got: {bfile}"
        )
    out: list[tuple[int, str]] = []
    missing: list[str] = []
    for chrom in range(1, 23):
        prefix = s.replace("@", str(chrom))
        if not Path(prefix + ".bed").exists():
            missing.append(prefix + ".bed")
        out.append((chrom, prefix))
    if missing:
        raise FileNotFoundError(
            f"Missing .bed files for {len(missing)} chromosome(s): {missing[:3]}..."
        )
    return out


def split_csv_arg(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def sub_chr(prefix: str, chrom: int) -> str:
    if "@" not in prefix:
        prefix = prefix + "@"
    return prefix.replace("@", str(chrom))


def _resolve_with_compression(base_path: str) -> tuple[str, str | None]:
    if Path(base_path + ".bz2").exists():
        return base_path + ".bz2", "bz2"
    if Path(base_path + ".gz").exists():
        return base_path + ".gz", "gzip"
    if Path(base_path).exists():
        return base_path, None
    raise FileNotFoundError(f"Cannot open {base_path}[.gz/.bz2]")


def get_present_chrs(prefix: str, max_chr: int = 22) -> list[int]:
    chrs = []
    for chrom in range(1, max_chr + 1):
        base = sub_chr(prefix, chrom)
        if glob.glob(base + ".*"):
            chrs.append(chrom)
    return chrs


def read_ldscore_chr(prefix: str, max_chr: int = 22) -> pd.DataFrame:
    """
    Mimic LDSC --ref-ld-chr / --w-ld-chr:
    reads {prefix}{chr}.l2.ldscore[.gz/.bz2] (or @ substitution) across chromosomes.
    """
    chrs = get_present_chrs(prefix, max_chr=max_chr)
    if not chrs:
        raise FileNotFoundError(f"No chromosome-split files found for prefix: {prefix}")

    frames = []
    for chrom in chrs:
        base = sub_chr(prefix, chrom) + ".l2.ldscore"
        path, comp = _resolve_with_compression(base)
        df = pd.read_csv(path, sep=r"\s+", compression=comp)
        if "MAF" in df.columns:
            df = df.drop(columns=["MAF"], errors="ignore")
        if "CM" in df.columns:
            df = df.drop(columns=["CM"], errors="ignore")
        frames.append(df)

    out = pd.concat(frames, axis=0, ignore_index=True)
    if "CHR" in out.columns and "BP" in out.columns:
        out = out.sort_values(["CHR", "BP"])
    if "SNP" not in out.columns:
        raise ValueError("LD score file missing SNP column.")
    out = out.drop(columns=["CHR", "BP"], errors="ignore")
    out = out.drop_duplicates(subset="SNP")
    return out


def read_ldscore_chr_multi(prefixes: str, max_chr: int = 22) -> pd.DataFrame:
    """
    Comma-separated prefixes, similar to LDSC ldscore_fromlist behavior.
    """
    p_list = split_csv_arg(prefixes)
    if len(p_list) == 1:
        return read_ldscore_chr(p_list[0], max_chr=max_chr)

    merged = None
    for i, p in enumerate(p_list):
        df = read_ldscore_chr(p, max_chr=max_chr)
        rename = {c: f"{c}_{i}" for c in df.columns if c != "SNP"}
        df = df.rename(columns=rename)
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="SNP", how="inner")
    if merged is None:
        raise ValueError("No LD score prefixes provided.")
    return merged


def read_frq_chr(prefix: str, max_chr: int = 22) -> pd.DataFrame:
    """
    Mimic LDSC --frqfile-chr:
    reads {prefix}{chr}.frq[.gz/.bz2] across chromosomes.
    """
    chrs = get_present_chrs(prefix, max_chr=max_chr)
    if not chrs:
        raise FileNotFoundError(f"No chromosome-split frq files found for prefix: {prefix}")

    frames = []
    for chrom in chrs:
        base = sub_chr(prefix, chrom) + ".frq"
        path, comp = _resolve_with_compression(base)
        df = pd.read_csv(path, sep=r"\s+", compression=comp)
        if "MAF" in df.columns and "FRQ" not in df.columns:
            df = df.rename(columns={"MAF": "FRQ"})
        if "FRQ" not in df.columns or "SNP" not in df.columns:
            raise ValueError(f"FRQ file missing SNP/FRQ columns: {path}")
        frames.append(df[["SNP", "FRQ"]])

    out = pd.concat(frames, axis=0, ignore_index=True).drop_duplicates(subset="SNP")
    return out
