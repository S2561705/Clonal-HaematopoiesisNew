"""
test_quantile.py
================
Confirm Fix 2: KI_3 refine is deterministic (identical across two runs)
and that the JAX core agrees with its NumPy twin (validate_participant).
"""

import sys
sys.path.append("..")
import warnings
warnings.filterwarnings("ignore")

import pickle as pk
import numpy as np

from src.KI_3 import (
    refine_optimal_model_posterior_vec,
    validate_participant,
    check_node_stability,
)

TARGET_PID  = "MDS711P64"
FITTED_FILE = "../exports/MDS/MDS_cohort_fitted.pk"
REFINE_S    = 40
REFINE_H    = 6

with open(FITTED_FILE, "rb") as f:
    cohort = pk.load(f)

part_template = next(p for p in cohort if p.uns.get("participant_id") == TARGET_PID)
cs = list(part_template.uns["model_dict"].values())[0][0]
print(f"Using known clonal structure: {cs}\n")

cols = ["fitness", "fitness_5", "fitness_95",
        "homozygosity", "homozygosity_5", "homozygosity_95"]


def run(label):
    part = part_template.copy()
    part.uns["model_dict"] = {"model_0": (cs, None)}
    part = refine_optimal_model_posterior_vec(
        part, s_resolution=REFINE_S, h_resolution=REFINE_H)
    print(f"--- {label} ---")
    print(part.obs[cols].to_string())
    print()
    return part.obs[cols].copy()


a = run("KI_3 run A")
b = run("KI_3 run B")

# MAP columns must match exactly (CIs use a fixed RNG seed, so those too)
diff = (a.to_numpy() - b.to_numpy())
print(f"max |A−B| across all reported columns: {np.nanmax(np.abs(diff)):.3e}")
print("IDENTICAL" if np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True) else "DIFFER")

print("\n=== validate_participant (JAX vs NumPy twin) ===")
validate_participant(part_template)

print("\n=== check_node_stability ===")
check_node_stability(part_template)
