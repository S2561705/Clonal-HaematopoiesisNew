"""
node_magnitude_check.py
========================
Cheap check of whether quadrature nodes x blow up relative to total_cells
near the h=1.0 pole, WITHOUT running the full HMM likelihood -- avoids the
OOM that check_qeps_stability hits on some participants at high resolution,
so this can isolate the pole-blowup question from ordinary R x R memory
scaling.
"""
import sys
sys.path.append("..")
import warnings
warnings.filterwarnings("ignore")

import pickle as pk
from src.KI_3 import check_node_magnitudes

TARGET_PID  = "MDS1134R53"
FITTED_FILE = "../exports/MDS/MDS_cohort_fitted.pk"

with open(FITTED_FILE, "rb") as f:
    cohort = pk.load(f)

part = next(p for p in cohort if p.uns.get("participant_id") == TARGET_PID)
print(f"Participant: {TARGET_PID}\n")

check_node_magnitudes(part, beta_resolution=500)