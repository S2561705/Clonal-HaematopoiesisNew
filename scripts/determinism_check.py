import sys
sys.path.append("..")
import pickle as pk
import copy
import numpy as np
from src.KI_3 import compute_clonal_models_prob_vec, refine_optimal_model_posterior_vec

with open("../exports/MDS/MDS_cohort_fitted.pk", "rb") as f:
    cohort = pk.load(f)
part = next(p for p in cohort if p.uns.get("participant_id") == "MDS711P64")

p1 = compute_clonal_models_prob_vec(copy.deepcopy(part))
p1 = refine_optimal_model_posterior_vec(p1)
p2 = compute_clonal_models_prob_vec(copy.deepcopy(part))
p2 = refine_optimal_model_posterior_vec(p2)

print("fitness p1:", p1.obs['fitness'].values)
print("fitness p2:", p2.obs['fitness'].values)
print("fitness match:", np.array_equal(p1.obs['fitness'].values, p2.obs['fitness'].values))
print("posterior grid match:", np.array_equal(
    np.array(p1.uns['optimal_model']['posterior']),
    np.array(p2.uns['optimal_model']['posterior'])))