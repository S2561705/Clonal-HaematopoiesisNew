import sys
sys.path.append("..")
import pickle as pk
import numpy as np

INPUT_FILE = "../exports/MDS/MDS_cohort_processed.pk"
TARGET_PID = "MDS711P64"   # swap for whichever failing participant you want to check
                             # (e.g. Participant 1/2/3 in your run's PID order)

with open(INPUT_FILE, "rb") as f:
    cohort = pk.load(f)

part = next(p for p in cohort if p.uns.get("participant_id") == TARGET_PID)

AO = part.layers['AO'].T.astype(float)
DP = part.layers['DP'].T.astype(float)
vaf = AO / np.maximum(DP, 1.0)

print(f"participant: {TARGET_PID}\n")
print(part.obs[['GENE']])
print()
print("max VAF per mutation (across all timepoints):")
for i, gene in enumerate(part.obs['GENE']):
    print(f"  {gene}: max_vaf={np.nanmax(vaf[:, i]):.3f}")