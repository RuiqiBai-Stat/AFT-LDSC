# AFT-LDSC (Minimal, LDSC-style IO)

This repository provides a minimal AFT-LDSC implementation for fast method verification.
Design goals: easy to read, easy to run, minimal dependencies.

The package has one CLI entry with four modes:

1. `sumstats`: fit AFT model and generate LDSC-style summary statistics for one phenotype
2. `heritability`: estimate total SNP heritability from one trait
3. `enrichment`: estimate partitioned heritability / enrichment
4. `gcor`: estimate genetic covariance and genetic correlation between two traits

## Quick Start (Clone and Run)

```bash
git clone <your-github-url>/AFT_LDSC.git
cd AFT_LDSC
python -m pip install -U pip
python -m pip install -e .
```

After installation, you can call either:

```bash
aft-ldsc --mode <sumstats|heritability|enrichment|gcor> --out <prefix> ...
```

or

```bash
python -m aft_ldsc.core --mode <sumstats|heritability|enrichment|gcor> --out <prefix> ...
```

## Example Tutorial (Ready-to-run)

The repository includes example trait files:

- `examples/250.2.sumstats.gz`
- `examples/250.2.alpha`
- `examples/401.1.sumstats.gz`
- `examples/401.1.alpha`

A runnable tutorial script is provided:

- `examples/tutorial.py`

It runs the following analyses and writes outputs to `examples/result/`:

1. heritability with LDSC LD panel
2. heritability with baselineLD panel
3. enrichment with baselineLD
4. enrichment with baselineLD + cell type group 1
5. genetic correlation (`gcor`) with LDSC LD panel

Run it with:

```bash
python examples/tutorial.py
```

You can override defaults by environment variables:

- `AFT_LDSC_LDPANEL`: local LD panel root path (default in script: `/Users/ruiqi/Desktop/survivalGWAS/LDpanel`)
- `AFT_LDSC_N_BLOCKS`: jackknife blocks used by tutorial commands (default: `200`)

## LD Panel Download Note

LD panel files are large and are not bundled in this repository.
Please download required LDSC resources to local disk from:

- https://console.cloud.google.com/storage/browser/broad-alkesgroup-public-requester-pays/LDSCORE?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))

Expected local resources for the tutorial include:

- `eur_w_ld_chr/`
- `1000G_Phase3_baselineLD_v2.2_ldscores/`
- `1000G_Phase3_weights_hm3_no_MHC/`
- `1000G_Phase3_frq/`
- `1000G_Phase3_cell_type_groups/`

## Method Flow (High-level)

### 1) Build survival summary statistics (`--mode sumstats`)

- Input: one phenotype (time + event), optional covariates, and chromosome-split PLINK `--bfile` template with an `@` placeholder
- Fit candidate AFT distributions (`lognormal`, `loglogistic`, `weibull`)
- Select model by weighted CvM GOF score
- Compute per-SNP statistic (stored as `Z`)
- Save:
  - `OUT.sumstats.gz` with columns: `SNP, N, Z, A1, A2`
  - `OUT.alpha` with one numeric alpha value (censoring factor)
  - `OUT.gof.tsv` and `OUT.log.tsv`

### 2) Estimate heritability (`--mode heritability`)

- Input:
  - `--sumstats <trait.sumstats.gz>`
  - `--alpha <trait.alpha>`
  - LD prefixes (`--ref-ld-chr`, and in baselineLD mode also `--w-ld-chr`, `--frqfile-chr`)
- Model uses `Z^2` as response with weighted regression
- Block jackknife provides uncertainty
- Save: `OUT.results.tsv`, `OUT.log`

### 3) Estimate enrichment (`--mode enrichment`)

- Input:
  - same `--sumstats`, `--alpha`
  - partitioned LD (`--ref-ld-chr`)
  - regression weights (`--w-ld-chr`)
  - frequency files (`--frqfile-chr`)
- Output includes:
  - `Category, Prop._SNPs, Prop._h2, Enrichment, Coefficient` (+ SE/P)
- Save:
  - `OUT.results.tsv`
  - `OUT.summary.tsv`
  - `OUT.log`

### 4) Estimate genetic correlation (`--mode gcor`)

- Input:
  - `--sumstats trait1.sumstats.gz,trait2.sumstats.gz`
  - `--alpha trait1.alpha,trait2.alpha`
  - LDSC-style LD prefix
- Performs allele alignment (flip sign when A1/A2 are swapped)
- Estimates `h2_1`, `h2_2`, `gcov`, `gcor`
- Save: `OUT.results.tsv`, `OUT.log`

## LDSC-style LD Input Convention

This implementation follows LDSC split-chromosome prefix style:

- `--ref-ld-chr <prefix>`
- `--w-ld-chr <prefix>`
- `--frqfile-chr <prefix>`

For each chromosome, files are resolved as:

- `prefix{chr}.l2.ldscore(.gz/.bz2)` for LD files
- `prefix{chr}.frq(.gz/.bz2)` for frequency files

If prefix contains `@`, `@` is replaced by chromosome number.
If prefix does not contain `@`, chromosome number is appended to the end.

Examples:

- `--ref-ld-chr ld/` reads `ld/1.l2.ldscore.gz ... ld/22.l2.ldscore.gz`
- `--ref-ld-chr ld/@_eur` reads `ld/1_eur.l2.ldscore.gz ... ld/22_eur.l2.ldscore.gz`

## End-to-end CLI Examples

### A) Generate one trait sumstats

```bash
python -m aft_ldsc.core \
  --mode sumstats \
  --input data/traitA_pheno.tsv \
  --event data/traitA_event.tsv \
  --covar data/covar.tsv \
  --bfile 'data/ukb_imp_chr@_clean' \
  --nproc 32 \
  --out out/traitA
```

### B) Heritability

```bash
python -m aft_ldsc.core \
  --mode heritability \
  --sumstats out/traitA.sumstats.gz \
  --alpha out/traitA.alpha \
  --ref-ld-chr data/eur_w_ld_chr/ \
  --out out/traitA_h2
```

### C) Enrichment

```bash
python -m aft_ldsc.core \
  --mode enrichment \
  --sumstats out/traitA.sumstats.gz \
  --alpha out/traitA.alpha \
  --ref-ld-chr data/baselineLD_v2.2/baselineLD. \
  --w-ld-chr data/weights_hm3_noMHC/weights.hm3_noMHC. \
  --frqfile-chr data/1000G.EUR.QC. \
  --out out/traitA_enrich
```

### D) Genetic correlation

```bash
python -m aft_ldsc.core \
  --mode gcor \
  --sumstats out/traitA.sumstats.gz,out/traitB.sumstats.gz \
  --alpha out/traitA.alpha,out/traitB.alpha \
  --ref-ld-chr data/eur_w_ld_chr/ \
  --out out/traitA_traitB_rg
```

## Dependencies

Core dependencies are listed in both `pyproject.toml` and `requirements.txt`:

- numpy
- pandas
- scipy
- statsmodels
- lifelines
- tqdm
- pysnptools

## Notes

- `sumstats` streams all 22 chromosomes in one pass and parallelizes per-SNP computation across SNP blocks (`--nproc`, default: `SLURM_CPUS_PER_TASK` or `1`).
- This repository prioritizes reproducible end-to-end usability for reviewers.
