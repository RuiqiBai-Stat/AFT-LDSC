#!/bin/bash
#SBATCH --job-name=aft_sumstats
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=test_individual/slurm_logs/%x_%j.out
#SBATCH --error=test_individual/slurm_logs/%x_%j.err

set -euo pipefail

PHECODE="${PHECODE:?PHECODE env var required}"
AFT_DIST="${AFT_DIST:?AFT_DIST env var required}"
BFILE="${BFILE:?BFILE env var required}"
OUT_PREFIX="${OUT_PREFIX:?OUT_PREFIX env var required}"
BLOCKSIZE="${BLOCKSIZE:-100}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate aft_ldsc

cd /work/users/b/r/brtang/surv_ldsc/AFT-LDSC

COVS="sex,$(python -c 'print(",".join([f"PC{i}" for i in range(1,41)]))')"

echo "Phecode: $PHECODE"
echo "AFT dist: $AFT_DIST"
echo "BFILE: $BFILE"
echo "OUT: $OUT_PREFIX"
echo "Blocksize: $BLOCKSIZE"
date

python -m aft_ldsc.core --mode sumstats \
  --input /work/users/r/u/ruiqi/survivalGWAS/yukang/survival_new.csv \
  --event /work/users/r/u/ruiqi/survivalGWAS/yukang/censor_new.csv \
  --covar /work/users/r/u/ruiqi/survivalLDSC/ukb_heri/real_data/2age_sex_inter_40gpc_more.txt \
  --bfile "$BFILE" \
  --censor-time /work/users/r/u/ruiqi/survivalLDSC/ukb_heri/real_data/censor_time_unrel.npy \
  --aft-dist "$AFT_DIST" \
  --iid-col IID --time-col "$PHECODE" --event-col "$PHECODE" \
  --covar-cols "$COVS" \
  --blocksize "$BLOCKSIZE" \
  --out "$OUT_PREFIX"

date
echo "Done."
