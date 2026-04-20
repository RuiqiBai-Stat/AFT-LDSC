# AFT-LDSC

AFT-LDSC is a command-line package for survival-trait LDSC-style analysis. It supports an end-to-end workflow from summary statistics generation to heritability, annotation enrichment, and genetic correlation.

Core modes:

1. `sumstats`: fit AFT model and generate LDSC-style summary statistics for one phenotype
2. `heritability`: estimate total SNP heritability from one trait
3. `enrichment`: estimate partitioned heritability / enrichment
4. `gcor`: estimate genetic covariance and genetic correlation between two traits

## Quick Start (Clone and Run)

```bash
git clone https://github.com/RuiqiBai-Stat/AFT-LDSC.git
cd AFT-LDSC
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

## Example Tutorial (Command-line)

Example trait files included in this repository:

- `examples/250.2.sumstats.gz`
- `examples/250.2.alpha`
- `examples/401.1.sumstats.gz`
- `examples/401.1.alpha`

A runnable shell tutorial is provided:

- `examples/tutorial.sh`

It runs and writes outputs to `examples/result/`:

1. heritability with LDSC LD panel
2. heritability with baselineLD v2.2 panel
3. enrichment with baselineLD v2.2
4. enrichment with baseline v1.2 + cell type group 1
5. genetic correlation (`gcor`) with LDSC LD panel

Run it with:

```bash
bash examples/tutorial.sh
```

Optional environment overrides:

- `AFT_LDSC_LDPANEL`: local LD panel root path
- `AFT_LDSC_N_BLOCKS`: jackknife blocks used by tutorial commands (default: `200`)

## LD Panel Download Note

LD panel files are large and are not bundled in this repository. Please download required LDSC resources to local disk from [https://console.cloud.google.com/storage/browser/broad-alkesgroup-public-requester-pays/LDSCORE](https://console.cloud.google.com/storage/browser/broad-alkesgroup-public-requester-pays/LDSCORE).

Expected local resources for the tutorial include:

- `eur_w_ld_chr/`
- `1000G_Phase3_baselineLD_v2.2_ldscores/`
- `baseline_v1.2/`
- `1000G_Phase3_weights_hm3_no_MHC/`
- `1000G_Phase3_frq/`
- `1000G_Phase3_cell_type_groups/`

## Method Flow (High-level)

### 1) Build survival summary statistics (`--mode sumstats`)

- Input: one phenotype (time + event), optional covariates, PLINK chromosome-split bfile template with an `@` placeholder (all 22 chromosomes are streamed in one pass)
- Fit candidate AFT distributions (`lognormal`, `loglogistic`, `weibull`)
- Select model by weighted CvM GOF score
- Compute per-SNP statistic (written as `Z` in sumstats)
- Save:
  - `OUT.sumstats.gz` with columns: `SNP, N, Z, A1, A2`
  - `OUT.alpha` with one numeric alpha value (censoring factor)
  - `OUT.gof.tsv` and `OUT.log.tsv`

### 2) Estimate heritability (`--mode heritability`)

- Input:
  - `--sumstats <trait.sumstats.gz>`
  - `--alpha <trait.alpha>`
  - LDSC-style LD prefixes (`--ref-ld-chr`; baseline mode also uses `--w-ld-chr`, `--frqfile-chr`)
- Model uses `Z^2` as response with weighted/iterative regression
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

The `--bfile` argument must contain a literal `@` placeholder (same
convention as `--ref-ld-chr` / `--w-ld-chr` / `--frqfile-chr`). The
tool replaces `@` with each chromosome 1..22 and streams all 22
`.bed/.bim/.fam` trios in one pass, writing a single combined
`.sumstats.gz`. An optional trailing `.bed` suffix on the argument is
accepted. All 22 files must exist on disk; if any are missing the run
aborts before fitting. The sample set used to fit the AFT model is the
intersection of IIDs across all 22 `.fam` files (and phenotype IIDs).
Quote the argument in shells where `@` may be interpreted.

The per-SNP GWAS loop is parallelized across SNP blocks with
`ProcessPoolExecutor`. `--nproc` controls the worker count; if omitted
it defaults to `SLURM_CPUS_PER_TASK` (else 1). `--blocksize` (default
100) sets SNPs per block. For best scaling pin BLAS to a single thread:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

A reference Slurm submission script is provided at
`test_individual/run_sumstats.sbatch` (32 CPUs, 64 GB, around 3-4 hours
for about 1.16 million HapMap3 SNPs and 350K samples).

Outputs:

- `out/traitA.sumstats.gz` — whole-genome, rows ordered chr1..chr22, sorted by BP within chromosome
- `out/traitA.alpha`
- `out/traitA.gof.tsv`
- `out/traitA.log.tsv`

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

## Why no real-data sumstats example

`sumstats` requires individual-level survival and genotype data. We do not upload real UK Biobank individual-level data as repository examples because of privacy and data-access restrictions. We provide summary statistics of 508 UK Biobank time-to-event phenotypes available for AFT-LDSC at [https://doi.org/10.5281/zenodo.19657574](https://doi.org/10.5281/zenodo.19657574).

In later versions, we plan to add a toy `sumstats` generation example based on public 1000 Genomes data.

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
