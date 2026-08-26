import pickle as pk
import numpy as np

with open("../exports/MDS/MDS_cohort_processed.pk", "rb") as f:
    cohort = pk.load(f)

print(f"loaded {len(cohort)} participants\n")

for p in cohort:
    pid = p.uns.get("participant_id")
    AO = np.array(p.layers['AO']); DP = np.array(p.layers['DP'])
    vaf = np.where(DP > 0, AO / np.where(DP > 0, DP, 1), np.nan)
    print(f"{pid}: {p.shape[0]} mutations, sex={p.uns.get('sex')}")
    for i in range(p.shape[0]):
        print(f"  idx {i}: {p.obs.index[i]:<22} "
              f"gene={str(p.obs['GENE'].iloc[i]):<8} "
              f"max_vaf={np.nanmax(vaf[i]):.3f}")
    print()