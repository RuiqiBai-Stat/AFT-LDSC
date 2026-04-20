"""Compare a combined whole-genome sumstats file against verify/."""
import sys
import numpy as np
import pandas as pd

phecode = sys.argv[1] if len(sys.argv) > 1 else "250.2"
path = f"test_individual/outputs/{phecode}.sumstats.gz"

ours = pd.read_csv(path, sep="\t")
ver = pd.read_csv(f"verify/{phecode}.sumstats.gz", sep="\t")

print(f"=== phecode {phecode}: ours n={len(ours)}, verify n={len(ver)} ===")

m = ours.merge(ver, on="SNP", suffixes=("_o", "_v"))
same = (m["A1_o"] == m["A1_v"]) & (m["A2_o"] == m["A2_v"])
flip = (m["A1_o"] == m["A2_v"]) & (m["A2_o"] == m["A1_v"])
other = int((~(same | flip)).sum())

z_o = m["Z_o"].to_numpy()
z_v = np.where(flip, -m["Z_v"].to_numpy(), m["Z_v"].to_numpy())
absd = np.abs(z_o - z_v)
r = np.corrcoef(z_o, z_v)[0, 1]
dn = (m["N_o"] - m["N_v"]).abs().max()

print(f"merged n={len(m)}, A1/A2 same={int(same.sum())}, "
      f"swapped={int(flip.sum())}, other={other}")
print(f"max|dZ|={absd.max():.3e}, mean|dZ|={absd.mean():.3e}, "
      f"Pearson r={r:.8f}, max|dN|={dn}")

ok = (absd.max() < 1e-8) and (r > 0.9999999) and (other == 0)
print(f"match: {ok}")
