import pickle as pk

with open('../exports/MDS/debug_participant_MDS581N49.pk', 'rb') as f:
    part = pk.load(f)

posterior = part.uns['optimal_model']['posterior']
s_range = part.uns['optimal_model']['s_range']

print("Posterior sum:", posterior.sum())
print("Posterior max:", posterior.max())
