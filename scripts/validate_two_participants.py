#!/usr/bin/env python
"""Run corrected validate_participant on the two diagnostic participants."""
import sys, pickle as pk
sys.path.append("..")
from src.KI_3 import validate_participant

INPUT_FILE = "../exports/MDS/MDS_cohort_processed.pk"
TARGET_PIDS = ["MDS711P64", "MDS1134R53", "MDS889H46"]

with open(INPUT_FILE, "rb") as f:
    cohort = pk.load(f)

by_pid = {p.uns.get("participant_id"): p for p in cohort}

for pid in TARGET_PIDS:
    part = by_pid.get(pid)
    if part is None:
        print(f"--- {pid}: NOT FOUND in cohort ---\n")
        continue
    print(f"--- {pid} ---")
    validate_participant(part, beta_resolution=1_000)   # match production
    print()
