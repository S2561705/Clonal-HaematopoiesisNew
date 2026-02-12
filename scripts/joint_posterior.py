import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# ======================
# 1. True parameters
# ======================
N_w = 1e5
read_depth = 1000
true_s = 0.25
true_h = 0.6
LOH_time = 20
sample_times = np.array([30, 40, 50])

# ======================
# 2. Clone growth models
# ======================
def het_clone(t, s):
    return np.exp(s * t)

def hom_clone(t, s, t_loh):
    return np.exp(s * np.maximum(t - t_loh, 0))

# ======================
# 3. Generate synthetic data
# ======================
x_het = het_clone(sample_times, true_s)
x_hom = hom_clone(sample_times, true_s, LOH_time)
het_cells = true_h * x_het
hom_cells = (1 - true_h) * x_hom
true_vaf = (het_cells + 2 * hom_cells) / (2 * (N_w + het_cells + hom_cells))

rng = np.random.default_rng(42)
AO_obs = rng.binomial(read_depth, true_vaf)

print(f"Observed AO: {AO_obs}")
print(f"Observed VAF: {AO_obs / read_depth}")

# ======================
# 4. PyMC inference model - FIXED VERSION
# ======================
with pm.Model() as model:
    # FIX 1: Narrower prior to prevent extreme values
    s = pm.Uniform("s", 0, 0.5)  # Changed from (0, 1)
    h = pm.Uniform("h", 0, 1)
    
    # FIX 2: Cap exponentials to prevent overflow
    MAX_EXP = 20  # exp(20) ≈ 5e8, manageable
    s_times = pm.math.minimum(s * sample_times, MAX_EXP)
    s_times_loh = pm.math.minimum(s * pm.math.maximum(sample_times - LOH_time, 0), MAX_EXP)
    
    x_het = pm.math.exp(s_times)
    x_hom = pm.math.exp(s_times_loh)
    
    het_cells = h * x_het
    hom_cells = (1 - h) * x_hom
    
    vaf = (het_cells + 2 * hom_cells) / (2 * (N_w + het_cells + hom_cells))
    
    pm.Binomial("AO", n=read_depth, p=vaf, observed=AO_obs)
    
    # FIX 3: Better sampling parameters
    trace = pm.sample(
        3000,
        tune=3000,
        target_accept=0.99,  # Changed from 0.95
        chains=4,  # Changed from 2
        random_seed=42,
        progressbar=True,
        init="adapt_diag"  # Better initialization
    )

# ======================
# 5. Check convergence
# ======================
print("\n=== Convergence Diagnostics ===")
summary = az.summary(trace, var_names=["s", "h"])
print(summary)

rhat = az.rhat(trace, var_names=["s", "h"])
print(f"\nR-hat values:")
print(rhat)
print(f"Max R-hat: {max(float(rhat['s']), float(rhat['h'])):.4f}")
print("(Should be < 1.01 for good convergence)")

divergences = trace.sample_stats.diverging.values.sum()
print(f"\nDivergences: {divergences}")
print("(Should be 0 or very few)")

# ======================
# 6. Posterior plots - FIXED ref_val format
# ======================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
az.plot_posterior(
    trace,
    var_names=["s", "h"],
    ref_val=[true_s, true_h],  # FIX 4: List instead of dict
    ax=axes
)
plt.tight_layout()
plt.show()

# ======================
# 7. Joint posterior
# ======================
axes = az.plot_pair(
    trace,
    var_names=["s", "h"],
    kind="kde",
    marginals=True,
)
# Diagonal (marginals)
axes[0, 0].axvline(true_s, color="red", ls="--", lw=2)
axes[1, 1].axvline(true_h, color="red", ls="--", lw=2)
# Off-diagonal (joint)
axes[1, 0].plot(true_s, true_h, "ro", markersize=8)
plt.show()

# ======================
# 8. VAF trajectory check
# ======================
def full_vaf(t, s, h):
    # Use same capping as in model
    MAX_EXP = 20
    s_times = np.minimum(s * t, MAX_EXP)
    s_times_loh = np.minimum(s * np.maximum(t - LOH_time, 0), MAX_EXP)
    
    x_het = np.exp(s_times)
    x_hom = np.exp(s_times_loh)
    
    het = h * x_het
    hom = (1 - h) * x_hom
    return (het + 2 * hom) / (2 * (N_w + het + hom))

t_grid = np.linspace(0, 60, 300)
true_curve = full_vaf(t_grid, true_s, true_h)

post_s = trace.posterior["s"].values.flatten()
post_h = trace.posterior["h"].values.flatten()

median_curve = full_vaf(
    t_grid,
    np.median(post_s),
    np.median(post_h),
)

plt.figure(figsize=(9, 5))
plt.plot(t_grid, true_curve, label="True VAF", lw=2)
plt.plot(t_grid, median_curve, "--", label="Inferred median VAF", lw=2)
plt.scatter(sample_times, AO_obs / read_depth, c="red", zorder=5, label="Observed")
plt.axvline(LOH_time, ls=":", c="gray", label="LOH time")
plt.xlabel("Time")
plt.ylabel("VAF")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("\n=== Summary ===")
print(f"True s = {true_s:.3f}, Posterior median = {np.median(post_s):.3f}")
print(f"True h = {true_h:.3f}, Posterior median = {np.median(post_h):.3f}")