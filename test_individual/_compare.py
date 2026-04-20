"""Compare chr22 sumstats against verify/."""
import sys
import numpy as np
import pandas as pd

phecode = sys.argv[1] if len(sys.argv) > 1 else "250.2"

ours = pd.read_csv(f"test_individual/outputs/{phecode}_chr22.sumstats.gz", sep="\t")
ver = pd.read_csv(f"verify/{phecode}.sumstats.gz", sep="\t")

# subset verify to chr22 SNPs
chr22_snps = set(ours["SNP"])
ver22 = ver[ver["SNP"].isin(chr22_snps)].copy()

print(f"=== {phecode} ===")
print(f"ours n={len(ours)}, verify n_all={len(ver)}, verify_chr22={len(ver22)}")

m = ours.merge(ver22, on="SNP", suffixes=("_o", "_v"))
print(f"merged n={len(m)}")

a_same = (m["A1_o"] == m["A1_v"]) & (m["A2_o"] == m["A2_v"])
a_flip = (m["A1_o"] == m["A2_v"]) & (m["A2_o"] == m["A1_v"])
print(f"A1/A2 same: {a_same.sum()}, swapped: {a_flip.sum()}, other: {(~(a_same|a_flip)).sum()}")

z_o = m["Z_o"].to_numpy()
z_v = m["Z_v"].to_numpy()
# align sign for swapped alleles
z_v_aligned = np.where(a_flip, -z_v, z_v)

diff = z_o - z_v_aligned
absd = np.abs(diff)
print(f"Z diff: max_abs={absd.max():.3e}, mean_abs={absd.mean():.3e}, "
      f"median_abs={np.median(absd):.3e}")
print(f"Z ours min/max: {z_o.min():.3f}/{z_o.max():.3f}")
print(f"Z verify min/max: {z_v_aligned.min():.3f}/{z_v_aligned.max():.3f}")

# correlation
r = np.corrcoef(z_o, z_v_aligned)[0, 1]
print(f"Pearson r(Z): {r:.6f}")

# N diff
n_diff = m["N_o"] - m["N_v"]
print(f"N diff: min={n_diff.min()}, max={n_diff.max()}, mean={n_diff.mean():.1f}")

# show top mismatches
m["absd"] = absd
top = m.nlargest(5, "absd")[["SNP", "A1_o", "A2_o", "A1_v", "A2_v",
                              "Z_o", "Z_v", "N_o", "N_v", "absd"]]
print("top |Z diff|:")
print(top.to_string(index=False))
