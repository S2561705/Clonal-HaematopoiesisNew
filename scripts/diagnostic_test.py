import pickle
with open('../exports/MDS/MDS_cohort_fitted_unified.pk', 'rb') as f:
    parts = pickle.load(f)

# Find MDS760G64
for part in parts:
    if part.uns['participant_id'] == 'MDS760G64':
        print("Time points:", part.var.time_points.values)
        print("\nAO (alternate allele counts):")
        print(part.layers['AO'])
        print("\nDP (total depth):")
        print(part.layers['DP'])
        print("\nVAFs (observed):")
        print(part.layers['AO'] / part.layers['DP'])
        print("\nInferred fitness:", part.obs['fitness'].values)
        break