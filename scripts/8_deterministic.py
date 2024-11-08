import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *
from src.clonal_inference import *
from src.deterministic_aux import *
import pickle

from multiprocessing import Pool

# Import results
with open('../exports/all_cohorts_fitted.pk', 'rb') as f:
    cohort = pk.load(f)

# drop participants with warnings
cohort = [part for part in cohort if part.uns['warning'] is None]

# drop MYC participant
cohort = [part for part in cohort if part.uns['participant_id']!='LBC360020']

for part in cohort:
    part.X = part.layers['AO']/part.layers['DP']

# Infer deterministic fit in parallel
with Pool(8) as p:
    processed_cohort = list(tqdm(
        p.imap(deterministic_fit, cohort),
        total=len(cohort)))

# Compute overall and within clone order of mutations
cohort = [order_of_mutations(part) 
    for part in tqdm(processed_cohort)]

# Export results
with open('../exports/all_processed_with_deterministic.pk', 'wb') as f:
    pk.dump(processed_cohort, f)
# %%
