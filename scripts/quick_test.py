import pickle as pk
import numpy as np
from scipy.stats import pearsonr

with open('../exports/MDS/MDS_cohort_processed.pk', 'rb') as f:
    cohort = pk.load(f)

part = cohort[1]  # MDS671W51
AO = part.layers['AO']
DP = part.layers['DP']
VAF = AO / np.maximum(DP, 1.0)

obs = list(part.obs.index)
for i in range(len(obs)):
    for j in range(i+1, len(obs)):
        r, _ = pearsonr(VAF[i], VAF[j])
        print(f"{obs[i][:15]} ↔ {obs[j][:15]}  r={r:.4f}")