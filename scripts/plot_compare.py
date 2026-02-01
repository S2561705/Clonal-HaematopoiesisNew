import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------
# Load fitted cohort
# ------------------------------
pk_path = '../exports/MDS/MDS_cohort_fitted.pk'
with open(pk_path, 'rb') as f:
    participant_list = pk.load(f)

print(f"Loaded {len(participant_list)} fitted participants from {pk_path}")

# ------------------------------
# Create output directory
# ------------------------------
output_dir = '../exports/MDS_fitness_plots'
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# Plot per participant
# ------------------------------
for i, part in enumerate(participant_list):
    participant_id = f"P{i+1}"

    # Check if obs has fitness info
    if 'fitness' not in part.obs.columns:
        print(f"No fitness info for participant {participant_id}, skipping.")
        continue

    df = part.obs.copy()
    
    plt.figure(figsize=(10,6))
    sns.set_style("whitegrid")

    for idx, row in df.iterrows():
        mutation_label = f"{row.get('PreferredSymbol', 'Mut')}:{row.get('HGVSc', '')}"
        s = row['fitness']
        s_low = row.get('fitness_5', s)
        s_high = row.get('fitness_95', s)
        
        # Plot point with error bars
        plt.errorbar(idx, s, yerr=[[s - s_low], [s_high - s]], fmt='o', label=mutation_label)

    plt.xlabel("Mutation index")
    plt.ylabel("Inferred fitness (s)")
    plt.title(f"Participant {participant_id}: Inferred fitness per mutation")
    plt.xticks(range(len(df)), [f"{row['PreferredSymbol']}\n{row['HGVSc']}" for _, row in df.iterrows()], rotation=45, ha='right')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{participant_id}_fitness.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved fitness plot for participant {participant_id} -> {plot_path}")

print("All participant fitness plots created!")



