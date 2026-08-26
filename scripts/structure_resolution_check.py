"""
Checks whether the WINNING clonal structure (not just its fitness/homozygosity
value) is stable as compute_clonal_models_prob_vec's resolution increases,
and -- critically -- how close the top-2 candidates' scores are at each
resolution. A flip between structures whose scores were already close is
consistent with genuine near-degeneracy; a flip between structures whose
scores looked confidently separated is a bug in the coarse-stage quadrature.
"""
import sys
sys.path.append("..")
import warnings; warnings.filterwarnings("ignore")
import pickle as pk
import copy
from src.KI_3 import compute_clonal_models_prob_vec

FITTED_FILE = "../exports/MDS/MDS_cohort_fitted.pk"
TARGET_PID = "MDS711P64"   # or whichever participant you saw this on

# (s_resolution, h_resolution, beta_resolution) triples to sweep

SETTINGS = [
    # sweep s_res / h_res at MODEST beta_res -- this is what tests structure stability
    (10, 3, 1000),
    (20, 4, 1000),   # current default
    (40, 6, 1000),
    (60, 8, 1000),
    # separately, confirm beta_res doesn't move the ranking, at LOW s_res
    (20, 4, 2000),
    (20, 4, 4000),
]

with open(FITTED_FILE, "rb") as f:
    cohort = pk.load(f)
part_template = next(p for p in cohort if p.uns.get("participant_id") == TARGET_PID)

print(f"participant: {TARGET_PID}\n")
print(f"{'s_res':>6} {'h_res':>6} {'beta_res':>9}  {'top structure':<30} "
      f"{'top score':>12}  {'2nd score':>12}  {'gap':>10}")

for s_res, h_res, beta_res in SETTINGS:
    part = copy.deepcopy(part_template)
    part = compute_clonal_models_prob_vec(
        part, s_resolution=s_res, h_resolution=h_res,
        beta_resolution=beta_res, disable_progressbar=True)
    ranked = list(part.uns['model_dict'].items())   # already sorted, best first
    top_cs, top_score = ranked[0][1]
    if len(ranked) > 1:
        second_cs, second_score = ranked[1][1]
        gap = top_score - second_score
        second_str = f"{second_score:.4f}"
    else:
        second_str, gap = "n/a", float('nan')
    print(f"{s_res:>6} {h_res:>6} {beta_res:>9}  {str(top_cs):<30} "
          f"{top_score:>12.4f}  {second_str:>12}  {gap:>10.4f}")