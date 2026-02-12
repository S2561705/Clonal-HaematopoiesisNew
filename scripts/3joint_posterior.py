import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import os

# Create output directory in current working directory
output_dir = "output_plots"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving plots to: {os.path.abspath(output_dir)}/\n")

# ======================
# 1. True parameters
# ======================
N_w = 1e5
read_depth = 1000
true_s = 0.25
true_h = 0.6
true_x0 = 100  # Initial clone size at t=0
LOH_time = 20
sample_times = np.array([30, 40, 50])

# ======================
# 2. Clone growth models (now with initial size)
# ======================
def het_clone(t, s, x0):
    """Heterozygous clone growth from initial size x0"""
    return x0 * np.exp(s * t)

def hom_clone(t, s, x0, t_loh):
    """Homozygous clone growth (starts at LOH time)"""
    return x0 * np.exp(s * np.maximum(t - t_loh, 0))

# ======================
# 3. Generate synthetic data
# ======================
x_het = het_clone(sample_times, true_s, true_x0)
x_hom = hom_clone(sample_times, true_s, true_x0, LOH_time)
het_cells = true_h * x_het
hom_cells = (1 - true_h) * x_hom
true_vaf = (het_cells + 2 * hom_cells) / (2 * (N_w + het_cells + hom_cells))

rng = np.random.default_rng(42)
AO_obs = rng.binomial(read_depth, true_vaf)

print("=== Data Generation ===")
print(f"True parameters: s={true_s}, h={true_h}, x0={true_x0}")
print(f"Sample times: {sample_times}")
print(f"True VAF: {true_vaf}")
print(f"Observed AO: {AO_obs}")
print(f"Observed VAF: {AO_obs / read_depth}\n")

# ======================
# 4. PyMC inference model - NOW WITH 3 PARAMETERS
# ======================
with pm.Model() as model:
    # Three parameters to infer
    s = pm.Uniform("s", 0, 0.5, initval=0.25)
    h = pm.Uniform("h", 0, 1, initval=0.6)
    x0 = pm.Uniform("x0", 1, 1000, initval=100)  # Initial clone size
    
    # Cap exponentials to prevent overflow
    MAX_EXP = 20
    s_times = pm.math.minimum(s * sample_times, MAX_EXP)
    s_times_loh = pm.math.minimum(s * pm.math.maximum(sample_times - LOH_time, 0), MAX_EXP)
    
    # Clone sizes (now scaled by x0)
    x_het = x0 * pm.math.exp(s_times)
    x_hom = x0 * pm.math.exp(s_times_loh)
    
    # Cell populations
    het_cells = h * x_het
    hom_cells = (1 - h) * x_hom
    
    # VAF calculation
    vaf = (het_cells + 2 * hom_cells) / (2 * (N_w + het_cells + hom_cells))
    
    # Likelihood
    pm.Binomial("AO", n=read_depth, p=vaf, observed=AO_obs)
    
    # Sample
    trace = pm.sample(
        3000,
        tune=3000,
        target_accept=0.99,
        chains=4,
        random_seed=42,
        progressbar=True,
        init="adapt_diag"
    )

# ======================
# 5. Convergence diagnostics
# ======================
print("=== Convergence Diagnostics ===")
summary = az.summary(trace, var_names=["s", "h", "x0"])
print(summary)

rhat = az.rhat(trace, var_names=["s", "h", "x0"])
print(f"\nR-hat values:")
print(rhat)
max_rhat = max(float(rhat['s']), float(rhat['h']), float(rhat['x0']))
print(f"Max R-hat: {max_rhat:.4f} (should be < 1.01)")

divergences = trace.sample_stats.diverging.values.sum()
print(f"Divergences: {divergences} (should be ~0)\n")

# ======================
# 6. 1D Marginal posteriors
# ======================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
az.plot_posterior(
    trace,
    var_names=["s", "h", "x0"],
    ref_val=[true_s, true_h, true_x0],
    ax=axes
)
plt.tight_layout()
plt.savefig(f"{output_dir}/posterior_marginals_3param.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir}/posterior_marginals_3param.png")

# ======================
# 7. 3x3 Joint Posterior Grid
# ======================
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# Get posterior samples
post_s = trace.posterior["s"].values.flatten()
post_h = trace.posterior["h"].values.flatten()
post_x0 = trace.posterior["x0"].values.flatten()

params = {
    's': post_s,
    'h': post_h,
    'x0': post_x0
}
param_names = ['s', 'h', 'x0']
true_values = {'s': true_s, 'h': true_h, 'x0': true_x0}

# Create 3x3 grid
for i, param_i in enumerate(param_names):
    for j, param_j in enumerate(param_names):
        ax = axes[i, j]
        
        if i == j:
            # DIAGONAL: 1D marginal histograms
            ax.hist(params[param_i], bins=50, density=True, alpha=0.7, color='steelblue')
            ax.axvline(true_values[param_i], color='red', ls='--', lw=2, label='True')
            ax.set_ylabel('Density')
            if i == 0:
                ax.set_title(param_i, fontsize=14, fontweight='bold')
            if i == 2:
                ax.set_xlabel(param_i, fontsize=12)
            ax.legend()
            
        elif i > j:
            # LOWER TRIANGLE: 2D scatter + KDE
            ax.scatter(params[param_j], params[param_i], 
                      alpha=0.1, s=1, color='steelblue', rasterized=True)
            
            # Add KDE contours
            from scipy.stats import gaussian_kde
            try:
                xy = np.vstack([params[param_j], params[param_i]])
                kde = gaussian_kde(xy)
                
                # Create grid for contour
                x_min, x_max = params[param_j].min(), params[param_j].max()
                y_min, y_max = params[param_i].min(), params[param_i].max()
                xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                zz = kde(positions).reshape(xx.shape)
                
                # Plot contours
                ax.contour(xx, yy, zz, levels=5, colors='black', alpha=0.3, linewidths=0.5)
            except:
                pass  # Skip KDE if it fails
            
            # Mark true values
            ax.plot(true_values[param_j], true_values[param_i], 
                   'r*', markersize=15, markeredgewidth=1.5, markeredgecolor='darkred')
            
            ax.set_ylabel(param_i, fontsize=12)
            if i == 2:
                ax.set_xlabel(param_j, fontsize=12)
                
        else:
            # UPPER TRIANGLE: 2D hexbin density
            hexbin = ax.hexbin(params[param_j], params[param_i], 
                              gridsize=30, cmap='Blues', mincnt=1, alpha=0.7)
            
            # Mark true values
            ax.plot(true_values[param_j], true_values[param_i], 
                   'r*', markersize=15, markeredgewidth=1.5, markeredgecolor='darkred')
            
            if i == 0:
                ax.set_title(param_j, fontsize=14, fontweight='bold')
            ax.set_ylabel(param_i, fontsize=12)
            if j == 2:
                plt.colorbar(hexbin, ax=ax, label='Count')

plt.tight_layout()
plt.savefig(f"{output_dir}/joint_posterior_3x3.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir}/joint_posterior_3x3.png")

# ======================
# 8. Alternative: ArviZ pair plot (cleaner but less customizable)
# ======================
axes_pair = az.plot_pair(
    trace,
    var_names=["s", "h", "x0"],
    kind="kde",
    marginals=True,
    figsize=(12, 12),
    divergences=True,
    textsize=12
)

# Add true values to marginals (diagonal)
axes_pair[0, 0].axvline(true_s, color="red", ls="--", lw=2, label="True s")
axes_pair[0, 0].legend()
axes_pair[1, 1].axvline(true_h, color="red", ls="--", lw=2, label="True h")
axes_pair[1, 1].legend()
axes_pair[2, 2].axvline(true_x0, color="red", ls="--", lw=2, label="True x0")
axes_pair[2, 2].legend()

# Add true values to joint plots (off-diagonal)
axes_pair[1, 0].plot(true_s, true_h, "ro", markersize=10, label="True values")
axes_pair[1, 0].legend()
axes_pair[2, 0].plot(true_s, true_x0, "ro", markersize=10)
axes_pair[2, 1].plot(true_h, true_x0, "ro", markersize=10)

plt.tight_layout()
plt.savefig(f"{output_dir}/joint_posterior_3x3_arviz.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir}/joint_posterior_3x3_arviz.png")

# ======================
# 9. Correlation matrix
# ======================
import pandas as pd

# Create correlation matrix
posterior_df = pd.DataFrame({
    's': post_s,
    'h': post_h,
    'x0': post_x0
})
corr_matrix = posterior_df.corr()

print("\n=== Posterior Correlation Matrix ===")
print(corr_matrix)

# Plot correlation matrix
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

# Add text annotations
for i in range(len(param_names)):
    for j in range(len(param_names)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=14)

ax.set_xticks(range(len(param_names)))
ax.set_yticks(range(len(param_names)))
ax.set_xticklabels(param_names, fontsize=12)
ax.set_yticklabels(param_names, fontsize=12)
ax.set_title("Posterior Correlation Matrix", fontsize=14, fontweight='bold')

plt.colorbar(im, ax=ax, label='Correlation')
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir}/correlation_matrix.png")

# ======================
# 10. Trace plots
# ======================
fig, axes = plt.subplots(3, 2, figsize=(12, 9))
az.plot_trace(trace, var_names=["s", "h", "x0"], axes=axes)
plt.tight_layout()
plt.savefig(f"{output_dir}/trace_plots_3param.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir}/trace_plots_3param.png")

# ======================
# 11. VAF trajectory with uncertainty
# ======================
def full_vaf(t, s, h, x0):
    MAX_EXP = 20
    s_times = np.minimum(s * t, MAX_EXP)
    s_times_loh = np.minimum(s * np.maximum(t - LOH_time, 0), MAX_EXP)
    
    x_het = x0 * np.exp(s_times)
    x_hom = x0 * np.exp(s_times_loh)
    
    het = h * x_het
    hom = (1 - h) * x_hom
    return (het + 2 * hom) / (2 * (N_w + het + hom))

t_grid = np.linspace(0, 60, 300)
true_curve = full_vaf(t_grid, true_s, true_h, true_x0)

# Sample posterior trajectories
n_samples = 500
vaf_samples = np.zeros((n_samples, len(t_grid)))
idx = np.random.choice(len(post_s), n_samples, replace=False)

for i, j in enumerate(idx):
    vaf_samples[i] = full_vaf(t_grid, post_s[j], post_h[j], post_x0[j])

# Compute percentiles
vaf_median = np.median(vaf_samples, axis=0)
vaf_95_low = np.percentile(vaf_samples, 2.5, axis=0)
vaf_95_high = np.percentile(vaf_samples, 97.5, axis=0)
vaf_50_low = np.percentile(vaf_samples, 25, axis=0)
vaf_50_high = np.percentile(vaf_samples, 75, axis=0)

plt.figure(figsize=(10, 6))
plt.fill_between(t_grid, vaf_95_low, vaf_95_high, alpha=0.2, color='C1', label='95% CI')
plt.fill_between(t_grid, vaf_50_low, vaf_50_high, alpha=0.3, color='C1', label='50% CI')
plt.plot(t_grid, true_curve, 'k-', label="True VAF", lw=2.5)
plt.plot(t_grid, vaf_median, 'C1--', label="Posterior median VAF", lw=2)
plt.scatter(sample_times, AO_obs / read_depth, c="red", s=100, zorder=5, 
            label="Observed", edgecolors='black', linewidths=1.5)
plt.axvline(LOH_time, ls=":", c="gray", lw=2, label="LOH time")
plt.xlabel("Time", fontsize=12)
plt.ylabel("VAF", fontsize=12)
plt.title("VAF Trajectory with Posterior Uncertainty", fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/vaf_trajectory_3param.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir}/vaf_trajectory_3param.png")

# ======================
# 12. Summary statistics
# ======================
print("\n=== Posterior Summary ===")
print(f"True s  = {true_s:.3f}, Posterior median = {np.median(post_s):.3f}")
print(f"  95% CI: [{np.percentile(post_s, 2.5):.3f}, {np.percentile(post_s, 97.5):.3f}]")

print(f"\nTrue h  = {true_h:.3f}, Posterior median = {np.median(post_h):.3f}")
print(f"  95% CI: [{np.percentile(post_h, 2.5):.3f}, {np.percentile(post_h, 97.5):.3f}]")

print(f"\nTrue x0 = {true_x0:.1f}, Posterior median = {np.median(post_x0):.1f}")
print(f"  95% CI: [{np.percentile(post_x0, 2.5):.1f}, {np.percentile(post_x0, 97.5):.1f}]")

# Check coverage
s_in_ci = np.percentile(post_s, 2.5) <= true_s <= np.percentile(post_s, 97.5)
h_in_ci = np.percentile(post_h, 2.5) <= true_h <= np.percentile(post_h, 97.5)
x0_in_ci = np.percentile(post_x0, 2.5) <= true_x0 <= np.percentile(post_x0, 97.5)

print(f"\nTrue s in 95% CI:  {s_in_ci}")
print(f"True h in 95% CI:  {h_in_ci}")
print(f"True x0 in 95% CI: {x0_in_ci}")

print("\n=== Analysis Complete ===")
print(f"All plots saved to: {os.path.abspath(output_dir)}/")
print("\nGenerated plots:")
print("  1. posterior_marginals_3param.png - 1D marginals for all 3 parameters")
print("  2. joint_posterior_3x3.png - Custom 3x3 grid (scatter + hexbin)")
print("  3. joint_posterior_3x3_arviz.png - ArviZ version (cleaner, with KDE)")
print("  4. correlation_matrix.png - Posterior correlations")
print("  5. trace_plots_3param.png - MCMC convergence")
print("  6. vaf_trajectory_3param.png - VAF predictions with uncertainty")