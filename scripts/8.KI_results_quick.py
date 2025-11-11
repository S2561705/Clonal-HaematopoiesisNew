import sys
sys.path.append("..")
from src.general_imports import *
import pickle as pk
import matplotlib.pyplot as plt
import seaborn as sns

# Load the fitted data
print("Loading fitted MDS cohort...")
with open('../exports/MDS/MDS_cohort_fitted.pk', 'rb') as f:
    fitted_cohort = pk.load(f)

print(f"Loaded {len(fitted_cohort)} participants")

# Set up the plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Plot fitness distribution across all mutations
all_fitness = []
all_genes = []
all_participants = []

for part in fitted_cohort:
    if 'fitness' in part.obs.columns:
        participant_id = part.uns.get('participant_id', 'Unknown')
        for idx, row in part.obs.iterrows():
            all_fitness.append(row['fitness'])
            all_genes.append(row['PreferredSymbol'])
            all_participants.append(participant_id)

# Create a DataFrame for easier plotting
fitness_df = pd.DataFrame({
    'fitness': all_fitness,
    'gene': all_genes,
    'participant': all_participants
})

print(f"Total mutations with fitness estimates: {len(fitness_df)}")

# Plot 1: Distribution of fitness values
plt.figure(figsize=(10, 6))
plt.hist(fitness_df['fitness'], bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Fitness')
plt.ylabel('Count')
plt.title('Distribution of Inferred Fitness Values')
plt.tight_layout()
plt.savefig('../exports/MDS/fitness_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Fitness by gene
plt.figure(figsize=(12, 6))
sns.boxplot(data=fitness_df, x='gene', y='fitness')
plt.xticks(rotation=45)
plt.xlabel('Gene')
plt.ylabel('Fitness')
plt.title('Fitness Estimates by Gene')
plt.tight_layout()
plt.savefig('../exports/MDS/fitness_by_gene.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Fitness by participant
plt.figure(figsize=(12, 6))
sns.boxplot(data=fitness_df, x='participant', y='fitness')
plt.xticks(rotation=45)
plt.xlabel('Participant')
plt.ylabel('Fitness')
plt.title('Fitness Estimates by Participant')
plt.tight_layout()
plt.savefig('../exports/MDS/fitness_by_participant.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 4: Scatter plot of fitness with confidence intervals for each participant
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, part in enumerate(fitted_cohort):
    if i >= len(axes):
        break
        
    if 'fitness' in part.obs.columns and 'fitness_5' in part.obs.columns and 'fitness_95' in part.obs.columns:
        participant_id = part.uns.get('participant_id', f'Participant_{i}')
        
        # Sort by fitness for better visualization
        sorted_indices = part.obs['fitness'].argsort()
        genes = part.obs['PreferredSymbol'].iloc[sorted_indices]
        fitness = part.obs['fitness'].iloc[sorted_indices]
        fitness_5 = part.obs['fitness_5'].iloc[sorted_indices]
        fitness_95 = part.obs['fitness_95'].iloc[sorted_indices]
        
        y_pos = range(len(genes))
        
        axes[i].errorbar(fitness, y_pos, xerr=[fitness - fitness_5, fitness_95 - fitness], 
                        fmt='o', capsize=5, alpha=0.7)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(genes)
        axes[i].set_xlabel('Fitness')
        axes[i].set_title(f'{participant_id}\n(n={len(genes)} mutations)')
        axes[i].grid(True, alpha=0.3)

# Remove empty subplots
for i in range(len(fitted_cohort), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig('../exports/MDS/fitness_with_CI_by_participant.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n=== Fitness Summary Statistics ===")
print(fitness_df.groupby('gene')['fitness'].describe())
print(f"\nOverall fitness statistics:")
print(fitness_df['fitness'].describe())

# Save the fitness data to CSV for further analysis
fitness_df.to_csv('../exports/MDS/fitness_results.csv', index=False)
print(f"\nFitness data saved to ../exports/MDS/fitness_results.csv")