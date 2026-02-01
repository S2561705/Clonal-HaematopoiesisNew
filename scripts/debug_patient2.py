# debug_patient2_with_plots.py
import sys
sys.path.append("..")
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

# Load the patient
with open('../exports/homozygous_test_cohort.pk', 'rb') as f:
    patients = pk.load(f)

patient = patients[2]  # Patient 2
print(f"Patient: {patient.uns['patient_id']}")
print(f"True structure: {patient.uns['true_clonal_structure']}")
print(f"True clone A (muts 0,1,2): {patient.uns['true_clonal_structure'][0]}")
print(f"True clone B (muts 3,4): {patient.uns['true_clonal_structure'][1]}")

# Extract data
time_points = patient.var.time_points
n_mutations = patient.shape[0]
AO = patient.layers['AO']
DP = patient.layers['DP']
vaf_matrix = AO / DP

# Create a figure with multiple subplots
fig = plt.figure(figsize=(16, 10))

# ============================================
# Plot 1: All VAF trajectories
# ============================================
ax1 = plt.subplot(2, 3, 1)
for i in range(n_mutations):
    ax1.plot(time_points, vaf_matrix[i], 'o-', linewidth=2, markersize=8, 
             label=f'Mutation {i}', alpha=0.8)

ax1.set_title('All VAF Trajectories', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time Point', fontsize=12)
ax1.set_ylabel('VAF', fontsize=12)
ax1.set_ylim(0, 1.1)
ax1.grid(True, alpha=0.3)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='VAF=1.0')

# ============================================
# Plot 2: VAFs colored by true clone
# ============================================
ax2 = plt.subplot(2, 3, 2)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
true_structure = patient.uns['true_clonal_structure']

for clone_idx, clone_muts in enumerate(true_structure):
    for mut_idx in clone_muts:
        ax2.plot(time_points, vaf_matrix[mut_idx], 'o-', linewidth=2, markersize=8,
                 color=colors[clone_idx % len(colors)], 
                 label=f'Clone {clone_idx}, Mut {mut_idx}' if mut_idx == clone_muts[0] else "")

ax2.set_title('VAFs Colored by TRUE Clone', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time Point', fontsize=12)
ax2.set_ylabel('VAF', fontsize=12)
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

# ============================================
# Plot 3: Correlation distance heatmap
# ============================================
ax3 = plt.subplot(2, 3, 3)

# Calculate correlation distances
correlation_matrix = np.corrcoef(np.vstack([patient.X, patient.var.time_points]))
correlation_vec = correlation_matrix[-1, :-1]
distance_matrix = np.abs(correlation_vec - correlation_vec[:, None])

# Create heatmap
im = ax3.imshow(distance_matrix, cmap='RdYlBu_r', interpolation='nearest')
plt.colorbar(im, ax=ax3, label='Correlation Distance')

# Add text annotations
for i in range(n_mutations):
    for j in range(n_mutations):
        text = ax3.text(j, i, f'{distance_matrix[i, j]:.2f}',
                       ha="center", va="center", 
                       color="white" if distance_matrix[i, j] > 0.5 else "black",
                       fontsize=9)

ax3.set_title('Correlation Distance Matrix', fontsize=14, fontweight='bold')
ax3.set_xlabel('Mutation Index', fontsize=12)
ax3.set_ylabel('Mutation Index', fontsize=12)
ax3.set_xticks(range(n_mutations))
ax3.set_yticks(range(n_mutations))

# Highlight invalid pairs (distance > 0.5)
invalid_pairs = []
for i in range(n_mutations):
    for j in range(n_mutations):
        if i < j and distance_matrix[i, j] > 0.5:
            invalid_pairs.append((i, j))
            # Draw red boxes
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                               edgecolor='red', linewidth=2)
            ax3.add_patch(rect)

# ============================================
# Plot 4: Hierarchical clustering dendrogram
# ============================================
ax4 = plt.subplot(2, 3, 4)

# Perform hierarchical clustering
condensed_dist = squareform(distance_matrix)
Z = linkage(condensed_dist, method='average')

# Create dendrogram
dendrogram(Z, labels=[f'Mut {i}' for i in range(n_mutations)],
           leaf_rotation=90, leaf_font_size=10,
           color_threshold=0.5, above_threshold_color='grey',
           ax=ax4)

ax4.set_title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
ax4.set_xlabel('Mutation', fontsize=12)
ax4.set_ylabel('Distance', fontsize=12)
ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold=0.5')
ax4.legend()

# ============================================
# Plot 5: VAF growth patterns (normalized)
# ============================================
ax5 = plt.subplot(2, 3, 5)

# Normalize VAFs to start at 0 for better comparison
vaf_normalized = vaf_matrix.copy()
for i in range(n_mutations):
    vaf_normalized[i] = (vaf_matrix[i] - vaf_matrix[i, 0]) / (vaf_matrix[i].max() - vaf_matrix[i, 0] + 1e-10)

for i in range(n_mutations):
    ax5.plot(time_points, vaf_normalized[i], 'o-', linewidth=2, markersize=6,
             label=f'Mutation {i}', alpha=0.8)

ax5.set_title('Normalized VAF Growth Patterns', fontsize=14, fontweight='bold')
ax5.set_xlabel('Time Point', fontsize=12)
ax5.set_ylabel('Normalized VAF', fontsize=12)
ax5.grid(True, alpha=0.3)
ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

# ============================================
# Plot 6: Summary statistics
# ============================================
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')  # Turn off axis for text display

# Calculate statistics
mean_vafs = np.mean(vaf_matrix, axis=1)
std_vafs = np.std(vaf_matrix, axis=1)
max_vafs = np.max(vaf_matrix, axis=1)
min_vafs = np.min(vaf_matrix, axis=1)

# Prepare text
summary_text = f"Patient: {patient.uns['patient_id']}\n"
summary_text += f"True Structure: {patient.uns['true_clonal_structure']}\n\n"

summary_text += "Mutation Statistics:\n"
for i in range(n_mutations):
    summary_text += f"  Mut {i}: mean={mean_vafs[i]:.3f}, std={std_vafs[i]:.3f}, "
    summary_text += f"range=[{min_vafs[i]:.3f}, {max_vafs[i]:.3f}]\n"

summary_text += f"\nInvalid Pairs (distance > 0.5): {len(invalid_pairs)}\n"
for i, (mut1, mut2) in enumerate(invalid_pairs[:6]):  # Show first 6
    dist = distance_matrix[mut1, mut2]
    in_same_clone = False
    for clone in true_structure:
        if mut1 in clone and mut2 in clone:
            in_same_clone = True
            break
    
    warning = "⚠️ SAME CLONE!" if in_same_clone else ""
    summary_text += f"  {mut1} & {mut2}: dist={dist:.3f} {warning}\n"

if len(invalid_pairs) > 6:
    summary_text += f"  ... and {len(invalid_pairs) - 6} more pairs\n"

# Add text to plot
ax6.text(0.05, 0.95, summary_text, fontfamily='monospace', fontsize=10,
         verticalalignment='top', transform=ax6.transAxes,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================
# Add overall title and adjust layout
# ============================================
plt.suptitle(f'Patient 2 Diagnostic: {patient.uns["patient_id"]}\n'
             f'{n_mutations} mutations, {len(true_structure)} clones', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('../exports/patient2_diagnostic.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# Console output for debugging
# ============================================
print("\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

print(f"\nVAF Matrix (mutations × timepoints):")
print("      T0    T1    T2    T3")
for i in range(n_mutations):
    print(f"Mut {i}: {vaf_matrix[i].round(3)}")

print(f"\nMean VAFs:")
for i in range(n_mutations):
    print(f"  Mut {i}: {mean_vafs[i]:.3f}")

print(f"\nCorrelation with time vector:")
print(f"  {correlation_vec.round(3)}")

print(f"\nProblematic pairs (distance > 0.5) that are in SAME TRUE CLONE:")
for clone_muts in true_structure:
    for i in range(len(clone_muts)):
        for j in range(i+1, len(clone_muts)):
            mut1, mut2 = clone_muts[i], clone_muts[j]
            dist = distance_matrix[mut1, mut2]
            if dist > 0.5:
                print(f"  Mut {mut1} & {mut2}: distance = {dist:.3f}")

print(f"\nRecommendation:")
print("1. The correlation filter (threshold=0.5) is rejecting valid clonal structures")
print("2. Mutations 0,1,2 should be in same clone but have different correlation patterns")
print("3. Try: threshold=0.8 or disable filtering for n_mutations ≤ 8")

# ============================================
# Additional analysis: What would clustering produce?
# ============================================
print("\n" + "="*80)
print("CLUSTERING ANALYSIS")
print("="*80)

# Try different clustering thresholds
thresholds = [0.3, 0.5, 0.7, 0.8]
for threshold in thresholds:
    clusters = fcluster(Z, t=threshold, criterion='distance')
    
    # Convert to clonal structure format
    cluster_dict = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(i)
    
    inferred_structure = [list(cluster) for cluster in cluster_dict.values()]
    
    # Compare with true structure
    is_correct = (sorted([sorted(clone) for clone in inferred_structure]) == 
                  sorted([sorted(clone) for clone in true_structure]))
    
    print(f"\nThreshold = {threshold}:")
    print(f"  Inferred structure: {inferred_structure}")
    print(f"  Matches true: {is_correct}")
    print(f"  Number of clusters: {len(inferred_structure)}")