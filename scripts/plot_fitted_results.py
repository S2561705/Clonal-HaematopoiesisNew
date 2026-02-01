import pickle as pk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain

# -------------------------------------
# Load fitted cohort
# -------------------------------------
print("Loading fitted MDS cohort...")
with open('../exports/MDS/MDS_cohort_fitted.pk', 'rb') as f:
    participant_list = pk.load(f)

print(f"Loaded {len(participant_list)} fitted participants")

# -------------------------------------
# Build mutation-level summary table
# -------------------------------------
mutation_rows = []
for i, part in enumerate(participant_list):
    participant_id = f"P{i+1}"
    for idx, row in part.obs.iterrows():
        mutation_rows.append({
            'Participant': participant_id,
            'Mutation': row.get('PreferredSymbol', f"Mut{idx}"),
            'Fitness': row.get('fitness', np.nan),
            'Fitness_5th': row.get('fitness_5', np.nan),
            'Fitness_95th': row.get('fitness_95', np.nan),
            'Clone_ID': row.get('clonal_index', np.nan)
        })

mutation_table = pd.DataFrame(mutation_rows)
print("\nMutation-level fitness table (first 10 rows):")
print(mutation_table.head(10))

# Optionally display as a matplotlib table
plt.figure(figsize=(12, mutation_table.shape[0]*0.15))
plt.axis('off')
tbl = plt.table(cellText=mutation_table.head(20).values,  # show only first 20 for readability
                colLabels=mutation_table.columns,
                cellLoc='center',
                loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.auto_set_column_width(col=list(range(len(mutation_table.columns))))
plt.title("Mutation-level Fitness Summary (first 20 rows)", fontsize=14)
plt.show()

# -------------------------------------
# PLOT 1 — Posterior density curves for fitness
# -------------------------------------
plt.figure(figsize=(10,6))
for part in participant_list:
    if 'optimal_model' not in part.uns:
        continue
    
    model = part.uns['optimal_model']
    s_range = model['s_range']          # grid of s values
    posterior = model['posterior']      # shape: (n_s, n_clones)
    cs = model['clonal_structure']
    
    # normalize posterior per clone
    posterior_norm = posterior / posterior.sum(axis=0)
    
    # loop through clones
    for i, clone in enumerate(cs):
        label = ','.join([part.obs.iloc[idx]['PreferredSymbol'] for idx in clone])
        sns.lineplot(x=s_range, y=posterior_norm[:, i], alpha=0.5, label=label)

plt.xlabel("Fitness (s)")
plt.ylabel("Posterior density")
plt.title("Posterior density curves for clones across participants")
plt.tight_layout()
plt.show()


# -------------------------------------
# PLOT 2 — Gene frequencies (PreferredSymbol)
# -------------------------------------
if 'PreferredSymbol' in mutation_table.columns:
    gene_counts = mutation_table['Mutation'].value_counts()

    plt.figure(figsize=(10,6))
    gene_counts.head(20).plot(kind='bar')
    plt.title("Top 20 most frequent mutated genes")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# -------------------------------------
# PLOT 3 — Clonal structure complexity
# -------------------------------------
if 'Clone_ID' in mutation_table.columns:
    clone_sizes = mutation_table.groupby(['Participant','Clone_ID']).size().values
    
    plt.figure(figsize=(8,5))
    plt.hist(clone_sizes, bins=range(1, max(clone_sizes)+2))
    plt.title("Number of mutations per clone")
    plt.xlabel("Clonal complexity (# mutations)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# -------------------------------------
# PLOT 4 — Mutation co-occurrence heatmap
# -------------------------------------
all_genes = sorted(mutation_table['Mutation'].unique())
gene_index = {g:i for i,g in enumerate(all_genes)}
mat = np.zeros((len(all_genes), len(all_genes)))

for _, row in mutation_table.iterrows():
    clone_mutations = mutation_table[(mutation_table['Participant']==row['Participant']) & 
                                    (mutation_table['Clone_ID']==row['Clone_ID'])]
    genes = clone_mutations['Mutation'].tolist()
    for g1 in genes:
        for g2 in genes:
            if g1 != g2:
                mat[gene_index[g1], gene_index[g2]] += 1

plt.figure(figsize=(12,10))
plt.imshow(mat, aspect='auto', cmap='viridis')
plt.colorbar(label='Co-occurrence count')
plt.title("Mutation co-occurrence heatmap")
plt.xticks(range(len(all_genes)), all_genes, rotation=90)
plt.yticks(range(len(all_genes)), all_genes)
plt.tight_layout()
plt.show()
