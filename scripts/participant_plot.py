import sys
sys.path.append("..")
from src.general_imports import *
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
print("Loading MDS cohort data...")
with open('../exports/MDS/MDS_cohort_processed.pk', 'rb') as f:
    MDS_cohort = pk.load(f)

print(f"Loaded {len(MDS_cohort)} participants")

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]

# Plot each participant
for i, part in enumerate(MDS_cohort):
    participant_id = part.uns.get('participant_id', f'Patient_{i}')
    
    # Extract data
    AO = np.array(part.layers['AO'])
    DP = np.array(part.layers['DP'])
    time_points = part.var['time_points'].values
    mutation_names = list(part.obs.index)
    
    # Calculate VAF
    VAF = AO / np.where(DP > 0, DP, 1)  # Avoid division by zero
    
    print(f"\nParticipant: {participant_id}")
    print(f"Shape: {AO.shape} (mutations × timepoints)")
    print(f"Time points: {time_points}")
    print(f"Mutations: {mutation_names}")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: VAF trajectories over time
    ax1 = axes[0]
    for j, mut_name in enumerate(mutation_names):
        ax1.plot(time_points, VAF[j, :], 'o-', linewidth=2, markersize=8, label=mut_name)
    
    ax1.set_xlabel('Time Points', fontsize=12)
    ax1.set_ylabel('Variant Allele Frequency (VAF)', fontsize=12)
    ax1.set_title(f'{participant_id} - VAF Trajectories', fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Plot 2: VAF heatmap
    ax2 = axes[1]
    im = ax2.imshow(VAF, aspect='auto', cmap='Reds', vmin=0, vmax=1)
    
    # Set labels
    ax2.set_xticks(range(len(time_points)))
    ax2.set_xticklabels(time_points)
    ax2.set_xlabel('Time Points', fontsize=12)
    
    ax2.set_yticks(range(len(mutation_names)))
    ax2.set_yticklabels(mutation_names)
    ax2.set_ylabel('Mutations', fontsize=12)
    ax2.set_title(f'{participant_id} - VAF Heatmap', fontsize=14)
    
    # Add colorbar
    plt.colorbar(im, ax=ax2, label='VAF')
    
    # Add VAF values as text on heatmap
    for y in range(VAF.shape[0]):
        for x in range(VAF.shape[1]):
            ax2.text(x, y, f'{VAF[y, x]:.3f}', ha='center', va='center', 
                    color='white' if VAF[y, x] > 0.5 else 'black', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'../exports/MDS/{participant_id}_VAF_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"VAF Statistics for {participant_id}:")
    for j, mut_name in enumerate(mutation_names):
        print(f"  {mut_name}:")
        for t, time_point in enumerate(time_points):
            print(f"    Time {time_point}: VAF = {VAF[j, t]:.4f} ({AO[j, t]}/{DP[j, t]})")
        print(f"    Max VAF: {VAF[j, :].max():.4f}")
        print(f"    Min VAF: {VAF[j, :].min():.4f}")
        print(f"    VAF change: {VAF[j, -1] - VAF[j, 0]:+.4f}")
    
    print("-" * 50)

# Create a summary plot for all participants
print("\nCreating summary plot for all participants...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, part in enumerate(MDS_cohort):
    if i >= len(axes):
        break
        
    participant_id = part.uns.get('participant_id', f'Patient_{i}')
    AO = np.array(part.layers['AO'])
    DP = np.array(part.layers['DP'])
    time_points = part.var['time_points'].values
    mutation_names = list(part.obs.index)
    VAF = AO / np.where(DP > 0, DP, 1)
    
    ax = axes[i]
    for j, mut_name in enumerate(mutation_names):
        ax.plot(time_points, VAF[j, :], 'o-', linewidth=2, markersize=6, label=mut_name)
    
    ax.set_title(f'{participant_id} ({len(mutation_names)} muts)', fontsize=11)
    ax.set_xlabel('Time')
    ax.set_ylabel('VAF')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

# Hide empty subplots if any
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('VAF Trajectories - All Participants', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('../exports/MDS/all_participants_VAF_summary.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nAll plots saved to ../exports/MDS/")