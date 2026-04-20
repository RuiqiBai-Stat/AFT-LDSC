#!/usr/bin/env bash
set -euo pipefail

# AFT-LDSC runnable tutorial (command-line version)
#
# This script runs:
# 1) heritability with LDSC LD panel
# 2) heritability with baselineLD v2.2 panel
# 3) enrichment with baselineLD v2.2
# 4) enrichment with baseline v1.2 + cell type group 1
# 5) gcor with LDSC LD panel
#
# LD panel files are large and not bundled in this repository.
# Download to local disk from:
# https://console.cloud.google.com/storage/browser/broad-alkesgroup-public-requester-pays/LDSCORE?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$SCRIPT_DIR/result"
mkdir -p "$OUT_DIR"

SUMSTATS_250="$SCRIPT_DIR/250.2.sumstats.gz"
ALPHA_250="$SCRIPT_DIR/250.2.alpha"
SUMSTATS_401="$SCRIPT_DIR/401.1.sumstats.gz"
ALPHA_401="$SCRIPT_DIR/401.1.alpha"

LDPANEL_ROOT="${AFT_LDSC_LDPANEL:-/Users/ruiqi/Desktop/survivalGWAS/LDpanel}"
N_BLOCKS="${AFT_LDSC_N_BLOCKS:-200}"

LDSC_PREFIX="$LDPANEL_ROOT/eur_w_ld_chr/"
BASELINELD_PREFIX="$LDPANEL_ROOT/1000G_Phase3_baselineLD_v2.2_ldscores/baselineLD."
BASELINE_V12_PREFIX="$LDPANEL_ROOT/baseline_v1.2/baseline."
WEIGHT_PREFIX="$LDPANEL_ROOT/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC."
FRQ_PREFIX="$LDPANEL_ROOT/1000G_Phase3_frq/1000G.EUR.QC."
CELLTYPE1_PREFIX="$LDPANEL_ROOT/1000G_Phase3_cell_type_groups/cell_type_group.1."

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] Required file not found: $1" >&2
    exit 1
  fi
}

require_ld_prefix() {
  local p="$1"
  if [[ -f "${p}1.l2.ldscore" || -f "${p}1.l2.ldscore.gz" || -f "${p}1.l2.ldscore.bz2" ]]; then
    return 0
  fi
  echo "[ERROR] Cannot find chr1 LD score file for prefix: $p" >&2
  exit 1
}

require_frq_prefix() {
  local p="$1"
  if [[ -f "${p}1.frq" || -f "${p}1.frq.gz" || -f "${p}1.frq.bz2" ]]; then
    return 0
  fi
  echo "[ERROR] Cannot find chr1 FRQ file for prefix: $p" >&2
  exit 1
}

run_cmd() {
  echo "[RUN] $*"
  "$@"
}

require_file "$SUMSTATS_250"
require_file "$ALPHA_250"
require_file "$SUMSTATS_401"
require_file "$ALPHA_401"

require_ld_prefix "$LDSC_PREFIX"
require_ld_prefix "$BASELINELD_PREFIX"
require_ld_prefix "$BASELINE_V12_PREFIX"
require_ld_prefix "$WEIGHT_PREFIX"
require_ld_prefix "$CELLTYPE1_PREFIX"
require_frq_prefix "$FRQ_PREFIX"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

echo "LD panel root: $LDPANEL_ROOT"
echo "Jackknife blocks: $N_BLOCKS"

# 1) Heritability with LDSC panel
run_cmd python -m aft_ldsc.core \
  --mode heritability \
  --sumstats "$SUMSTATS_250" \
  --alpha "$ALPHA_250" \
  --ref-ld-chr "$LDSC_PREFIX" \
  --n-blocks "$N_BLOCKS" \
  --out "$OUT_DIR/250.2_h2_ldsc"

# 2) Heritability with baselineLD v2.2 panel
run_cmd python -m aft_ldsc.core \
  --mode heritability \
  --sumstats "$SUMSTATS_250" \
  --alpha "$ALPHA_250" \
  --ref-ld-chr "$BASELINELD_PREFIX" \
  --w-ld-chr "$WEIGHT_PREFIX" \
  --frqfile-chr "$FRQ_PREFIX" \
  --n-blocks "$N_BLOCKS" \
  --out "$OUT_DIR/250.2_h2_baselineLD"

# 3) Enrichment with baselineLD v2.2
run_cmd python -m aft_ldsc.core \
  --mode enrichment \
  --sumstats "$SUMSTATS_250" \
  --alpha "$ALPHA_250" \
  --ref-ld-chr "$BASELINELD_PREFIX" \
  --w-ld-chr "$WEIGHT_PREFIX" \
  --frqfile-chr "$FRQ_PREFIX" \
  --n-blocks "$N_BLOCKS" \
  --out "$OUT_DIR/250.2_enrichment_baselineLD"

# 4) Enrichment with baseline v1.2 + cell type group 1
run_cmd python -m aft_ldsc.core \
  --mode enrichment \
  --sumstats "$SUMSTATS_250" \
  --alpha "$ALPHA_250" \
  --ref-ld-chr "$BASELINE_V12_PREFIX,$CELLTYPE1_PREFIX" \
  --w-ld-chr "$WEIGHT_PREFIX" \
  --frqfile-chr "$FRQ_PREFIX" \
  --n-blocks "$N_BLOCKS" \
  --out "$OUT_DIR/250.2_enrichment_baselinev12_celltype1"

# 5) Genetic correlation with LDSC panel
run_cmd python -m aft_ldsc.core \
  --mode gcor \
  --sumstats "$SUMSTATS_250,$SUMSTATS_401" \
  --alpha "$ALPHA_250,$ALPHA_401" \
  --ref-ld-chr "$LDSC_PREFIX" \
  --n-blocks "$N_BLOCKS" \
  --out "$OUT_DIR/250.2_401.1_gcor_ldsc"

echo "[DONE] Tutorial finished. Results written to: $OUT_DIR"
