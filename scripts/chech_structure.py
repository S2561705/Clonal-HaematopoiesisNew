"""
Debug script for MDS671W51 - corrected interpretation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import pickle as pk

# Load data
with open('../exports/MDS/MDS_cohort_fitted_unified.pk', 'rb') as f:
    processed_parts = pk.load(f)

# Find MDS671W51
for part in processed_parts:
    if part.uns.get('participant_id') == 'MDS671W51':
        break

print("="*80)
print("MDS671W51 CLONAL STRUCTURE - CORRECTED ANALYSIS")
print("="*80)

# Extract data
VAF = part.layers['AO'] / np.maximum(part.layers['DP'], 1)
time_points = part.var.time_points.values
mutations = list(part.obs.index)

print("\nVAF Trajectories:")
for i, mut in enumerate(mutations):
    vaf_str = ' → '.join([f'{v:.3f}' for v in VAF[i]])
    trend = "GROWING" if VAF[i, -1] > VAF[i, 0] else "DECLINING" if VAF[i, -1] < VAF[i, 0] else "STABLE"
    print(f"  {mut:<25} {vaf_str:<25} [{trend}]")

print("\n" + "="*80)
print("PROPOSED CLONAL STRUCTURE (4 clones)")
print("="*80)

clones = [
    {
        'name': 'Clone 1 (Founder)',
        'mutations': [0],  # DNMT3A
        'description': 'High VAF (~42%), stable → likely HOMOZYGOUS (cn-LOH)',
        'h_expected': 1.0
    },
    {
        'name': 'Clone 2 (Co-dominant)',
        'mutations': [1, 2],  # SF3B1, TET2 c.2961C>A
        'description': 'High VAF (~35-44%), growing → likely HOMOZYGOUS (cn-LOH)',
        'h_expected': 1.0
    },
    {
        'name': 'Clone 3 (Expanding subclone)',
        'mutations': [3],  # TET2 c.550G>T
        'description': 'Low VAF (~2-6%), growing → HETEROZYGOUS',
        'h_expected': 0.0
    },
    {
        'name': 'Clone 4 (Declining/lost)',
        'mutations': [4],  # TET2 c.5618T>C
        'description': 'VAF declining (8% → 0.5%) → HETEROZYGOUS, being outcompeted',
        'h_expected': 0.0
    }
]

for clone in clones:
    print(f"\n{clone['name']}:")
    print(f"  Mutations: {[mutations[i] for i in clone['mutations']]}")
    print(f"  {clone['description']}")
    print(f"  Expected h: {clone['h_expected']:.1f}")
    
    # Show mean VAF
    clone_vafs = [VAF[i] for i in clone['mutations']]
    mean_vaf = np.mean(clone_vafs, axis=0)
    print(f"  Mean VAF: {' → '.join([f'{v:.3f}' for v in mean_vaf])}")

print("\n" + "="*80)
print("WHY INFERENCE FAILED")
print("="*80)

print("\nThe model found 1 clone with all 5 mutations, which is IMPOSSIBLE because:")
print("  1. TET2 c.5618T>C is DECLINING while others are stable/growing")
print("  2. Strong negative correlations (r≈-0.94) indicate opposing dynamics")
print("  3. Different VAF levels suggest different zygosity states")

print("\nThe correlation-based filtering should have caught this but didn't.")
print("Likely issues:")
print("  - Filter uses 'correlation with time' instead of 'pairwise VAF correlation'")
print("  - Threshold may be too permissive")

# Check what models were evaluated
print("\n" + "="*80)
print("MODELS THAT WERE EVALUATED")
print("="*80)

if 'model_dict' in part.uns:
    print(f"\nTotal models evaluated: {len(part.uns['model_dict'])}")
    
    # Look for 4-clone structures
    four_clone_models = []
    three_clone_models = []
    
    for model_name, model_data in part.uns['model_dict'].items():
        if isinstance(model_data, tuple) and len(model_data) == 2:
            cs, prob = model_data
            
            if len(cs) == 4:
                four_clone_models.append((model_name, cs, prob))
            elif len(cs) == 3:
                three_clone_models.append((model_name, cs, prob))
    
    print(f"\n4-clone structures evaluated: {len(four_clone_models)}")
    if len(four_clone_models) > 0:
        print("Top 4-clone models:")
        for name, cs, prob in sorted(four_clone_models, key=lambda x: x[2], reverse=True)[:3]:
            print(f"  {name}: prob={prob:.3e}, structure={cs}")
    else:
        print("  ⚠️  NO 4-clone structures were evaluated!")
    
    print(f"\n3-clone structures evaluated: {len(three_clone_models)}")
    if len(three_clone_models) > 0:
        print("Top 3-clone models:")
        for name, cs, prob in sorted(three_clone_models, key=lambda x: x[2], reverse=True)[:3]:
            print(f"  {name}: prob={prob:.3e}, structure={cs}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\nThis participant should be re-analyzed with:")
print("  1. Fixed correlation filtering (use pairwise VAF correlations)")
print("  2. Lower correlation threshold (~0.7) to separate clones")
print("  3. Manual specification of 4-clone structure: [[0], [1,2], [3], [4]]")
print("  4. Independent h inference for each clone")
print("\nExpected results:")
print("  Clone 1 (DNMT3A): s≈0.0, h≈1.0")
print("  Clone 2 (SF3B1+TET2): s≈0.1-0.2, h≈1.0")
print("  Clone 3 (TET2 c.550G>T): s≈0.2-0.3, h≈0.0")
print("  Clone 4 (TET2 c.5618T>C): s<0 (negative!), h≈0.0")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: VAF trajectories with proposed structure
ax1 = axes[0, 0]
colors = ['red', 'blue', 'green', 'orange']
for clone_idx, clone in enumerate(clones):
    for mut_idx in clone['mutations']:
        ax1.plot(time_points, VAF[mut_idx], 'o-', color=colors[clone_idx],
                linewidth=2.5, markersize=10, alpha=0.8,
                label=mutations[mut_idx])

ax1.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.4, label='Het max')
ax1.set_xlabel('Time (years)', fontsize=12, fontweight='bold')
ax1.set_ylabel('VAF', fontsize=12, fontweight='bold')
ax1.set_title('Proposed 4-Clone Structure', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)

# Panel 2: Correlation matrix
ax2 = axes[0, 1]
n_mut = len(mutations)
corr_matrix = np.zeros((n_mut, n_mut))
for i in range(n_mut):
    for j in range(n_mut):
        if i != j:
            corr, _ = pearsonr(VAF[i], VAF[j])
            corr_matrix[i, j] = corr
        else:
            corr_matrix[i, j] = 1.0

im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax2.set_xticks(range(n_mut))
ax2.set_yticks(range(n_mut))
ax2.set_xticklabels([f"{i}" for i in range(n_mut)], fontsize=10)
ax2.set_yticklabels([mut[:15] for mut in mutations], fontsize=8)
ax2.set_title('VAF Correlation Matrix', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

for i in range(n_mut):
    for j in range(n_mut):
        text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
        ax2.text(j, i, f'{corr_matrix[i, j]:.2f}',
                ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')

# Panel 3: LOH signature
ax3 = axes[1, 0]
mean_vafs = [np.mean(VAF[i]) for i in range(n_mut)]
clone_colors = []
for i in range(n_mut):
    for clone_idx, clone in enumerate(clones):
        if i in clone['mutations']:
            clone_colors.append(colors[clone_idx])
            break

bars = ax3.barh(mutations, mean_vafs, color=clone_colors, alpha=0.7, edgecolor='black', linewidth=2)
ax3.axvline(0.5, color='gray', linestyle='--', linewidth=2, label='Het max (50%)')
ax3.axvline(0.25, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Het at 50% clone')
ax3.set_xlabel('Mean VAF', fontsize=12, fontweight='bold')
ax3.set_title('LOH Signature Analysis', fontsize=14, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='x')

for i, (mut, vaf) in enumerate(zip(mutations, mean_vafs)):
    if vaf > 0.3:
        ax3.text(vaf + 0.02, i, 'cn-LOH', fontsize=9, va='center',
                color='darkred', fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Panel 4: Clone dynamics summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = """
CLONAL ARCHITECTURE SUMMARY

Clone 1 (Founder):
  • DNMT3A c.1792C>T
  • VAF ~42% (stable)
  • HOMOZYGOUS (cn-LOH)
  • Fitness: s ≈ 0

Clone 2 (Co-dominant):
  • SF3B1 c.2098A>G
  • TET2 c.2961C>A
  • VAF ~35-44% (growing)
  • HOMOZYGOUS (cn-LOH)
  • Fitness: s ≈ 0.1-0.2

Clone 3 (Expanding):
  • TET2 c.550G>T
  • VAF ~2-6% (growing)
  • HETEROZYGOUS
  • Fitness: s ≈ 0.2-0.3

Clone 4 (Declining):
  • TET2 c.5618T>C
  • VAF 8% → 0.5% (declining)
  • HETEROZYGOUS
  • Fitness: s < 0 (negative)
  • Being outcompeted/lost

KEY FINDINGS:
✓ 2 dominant clones with cn-LOH
✓ 1 expanding subclone
✓ 1 declining/lost clone
✗ Single-clone model is WRONG
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1))

plt.tight_layout()
plt.savefig('../exports/MDS/MDS671W51_corrected_structure.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: ../exports/MDS/MDS671W51_corrected_structure.png")

print("\n" + "="*80)