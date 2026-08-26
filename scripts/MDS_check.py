import sys
sys.path.append("..")
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = "../exports/MDS/MDS_cohort_processed.pk"
TARGET_PID = "MDS1134R53"   # swap for MDS581N49 / MDS711P64 to check the others

with open(INPUT_FILE, "rb") as f:
    cohort = pk.load(f)

part = next(p for p in cohort if p.uns.get("participant_id") == TARGET_PID)

AO = part.layers['AO'].T.astype(float)
DP = part.layers['DP'].T.astype(float)
vaf = AO / np.maximum(DP, 1.0)
tp = np.array(part.var['time_points'], dtype=float)

fig, ax = plt.subplots(figsize=(6, 4))
for i, gene in enumerate(part.obs['GENE']):
    obs = DP[:, i] > 0   # only plot actually-observed timepoints
    ax.plot(tp[obs], vaf[obs, i], 'o-', label=f"{gene} ({part.obs.index[i]})")

ax.set_xlabel("time")
ax.set_ylabel("VAF")
ax.set_title(f"{TARGET_PID}: VAF over time")
ax.legend()
ax.grid(alpha=0.3)

out_png = f"{TARGET_PID}_vaf_trajectory.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"saved -> {out_png}")
plt.show()