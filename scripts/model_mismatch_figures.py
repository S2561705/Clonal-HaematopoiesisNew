from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import binom

# Matplotlib setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------

@dataclass(frozen=True)
class SimConfig:
    K: float = 1e5          # Constant normal cell count
    t_end: float = 200.0  # Simulation end time
    t_points: int = 2000  # Time points for ODE
    dp_mean: float = 5000.0   # Mean sequencing depth
    dp_sd: float = 2000.0    # Std dev of depth
    dp_min: int = 100        # Minimum depth
    sample_every: int = 60   # Sample every N time points
    vaf_threshold: float = 0.05  # VAF detection threshold
    followup_window: float = 80.0  # Observation window after detection
    seed: int = 123

# -----------------------------
# Data-generating ODE model
# -----------------------------

def model_original_B(y, t, s, E, B):
    """New ODE model with competition parameter B."""
    x, n, F = y
    dxdt = s * x * (1.0 - ((x + n) / E))
    if (F + n) > 1.0:
        dndt = -dxdt * (n / ((F * B) + n))
        dFdt = -dxdt * (1.0 - (n / ((F * B) + n)))
    else:
        dndt = 0.0
        dFdt = 0.0
    return [dxdt, dndt, dFdt]


def run_model_new(fitness, x0, B, E, cfg: SimConfig):
    """Run the new ODE model."""
    n0 = float(cfg.K)
    F0 = float(E) - n0
    y0 = [float(x0), n0, F0]
    t = np.linspace(0.0, float(cfg.t_end), int(cfg.t_points))
    sol = odeint(model_original_B, y0, t, args=(float(fitness), float(E), float(B)))
    x = sol[:, 0]
    n = sol[:, 1]
    F = sol[:, 2]
    return t, x, n, F


# -----------------------------
# Synthetic observation generation
# -----------------------------

def generate_synth_new(fitness, x0, B, E, cfg: SimConfig):
    """Generate synthetic VAF observations from the new model."""
    rng = np.random.default_rng(int(cfg.seed))
    t, x, n, _F = run_model_new(fitness, x0, B, E, cfg)
    
    # Generate sequencing depths
    dp = rng.normal(loc=cfg.dp_mean, scale=cfg.dp_sd, size=len(t)).astype(int)
    dp = np.maximum(dp, int(cfg.dp_min))
    
    # True VAF probability
    p = x / (2.0 * (n + x))
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    
    # Observed alternate reads
    ao = rng.binomial(dp, p).astype(int)
    ao = np.maximum(ao, 1)
    
    vaf_obs = ao / dp
    
    # Subsample
    step = int(cfg.sample_every)
    idx = np.arange(0, len(t), step)
    return t[idx], vaf_obs[idx], dp[idx], ao[idx]


def apply_threshold_and_window(t, vaf, dp, ao, cfg: SimConfig):
    """Trim observations: start at VAF threshold, cap at follow-up window."""
    start_idxs = np.where(vaf >= float(cfg.vaf_threshold))[0]
    if start_idxs.size == 0:
        return None
    s0 = int(start_idxs[0])
    
    t2 = t[s0:]
    v2 = vaf[s0:]
    dp2 = dp[s0:]
    ao2 = ao[s0:]
    
    if cfg.followup_window is not None and np.isfinite(cfg.followup_window):
        end_idxs = np.where(t2 >= (t[s0] + float(cfg.followup_window)))[0]
        if end_idxs.size > 0:
            e0 = int(end_idxs[0])
            t2 = t2[:e0]
            v2 = v2[:e0]
            dp2 = dp2[:e0]
            ao2 = ao2[:e0]
    
    if len(t2) < 2:
        return None
    
    return t2, v2, dp2, ao2


# -----------------------------
# Inference: Simple exponential MAP
# -----------------------------

def _softplus(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return np.maximum(u, 0.0) + np.log1p(np.exp(-np.abs(u)))


def fit_simple_exponential_map(t, dp, ao, K=1e5, s_upper=1.0, n_starts=6, seed=0):
    """
    Fit misspecified simple exponential model:
      x(t) = x0 * exp(s * (t - t0))
      p(t) = x(t) / (2*(x(t)+K))
      AO ~ Binomial(DP, p(t))
    Returns MAP estimates for s and x0.
    """
    t = np.asarray(t, dtype=float)
    dp = np.asarray(dp, dtype=int)
    ao = np.asarray(ao, dtype=int)
    dt = t - float(t[0])
    
    K = float(K)
    s_upper = float(s_upper)
    rng = np.random.default_rng(int(seed))
    
    def unpack(theta):
        u, w = theta
        x0 = float(_softplus(u)) + 1e-12
        s = float(s_upper * expit(w))
        return x0, s
    
    def nll(theta):
        x0, s = unpack(theta)
        z = np.clip(s * dt, -700.0, 700.0)
        x_t = x0 * np.exp(z)
        p_t = x_t / (2.0 * (x_t + K))
        p_t = np.clip(p_t, 1e-12, 1.0 - 1e-12)
        return float(-np.sum(binom.logpmf(ao, dp, p_t)))
    
    best = None
    for i in range(int(n_starts)):
        theta0 = np.array([rng.normal(0.0, 2.0), rng.normal(0.0, 1.5)], dtype=float)
        if i == 0:
            theta0 = np.array([np.log(1.0), 0.0], dtype=float)
        res = minimize(nll, theta0, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    
    x0_hat, s_hat = unpack(best.x)
    return float(s_hat), float(x0_hat), best  # Return optimization result too


# =============================
# NEW: Validation & Accuracy Demonstration Functions
# =============================

def run_single_validation(
    E: float,
    B: float,
    s_true: float,
    x0_true: float,
    cfg: SimConfig,
    s_upper: float = 1.0,
    map_starts: int = 6,
    rep: int = 0,
) -> dict:
    """
    Run a single detailed validation case with full diagnostics.
    Returns comprehensive results including trajectory data.
    """
    rng_seed = int(cfg.seed) + rep
    cfg_r = SimConfig(**{**asdict(cfg), "seed": rng_seed})
    
    # Generate synthetic data
    t_full, vaf_full, dp_full, ao_full = generate_synth_new(
        fitness=s_true, x0=x0_true, E=E, B=B, cfg=cfg_r
    )
    
    # Apply threshold/window
    masked = apply_threshold_and_window(t_full, vaf_full, dp_full, ao_full, cfg_r)
    if masked is None:
        return {"status": "failed_threshold", "E": E, "B": B, "rep": rep}
    
    t_obs, vaf_obs, dp_obs, ao_obs = masked
    
    # Fit model
    try:
        s_hat, x0_hat, opt_result = fit_simple_exponential_map(
            t_obs, dp_obs, ao_obs,
            K=float(cfg_r.K),
            s_upper=float(s_upper),
            n_starts=int(map_starts),
            seed=int(cfg_r.seed),
        )
    except Exception as e:
        return {
            "status": f"fit_failed:{type(e).__name__}",
            "E": E, "B": B, "rep": rep,
            "n_points": len(t_obs),
        }
    
    # Compute inferred trajectory
    dt_obs = t_obs - t_obs[0]
    x_inferred = x0_hat * np.exp(s_hat * dt_obs)
    p_inferred = x_inferred / (2.0 * (x_inferred + cfg_r.K))
    vaf_inferred = p_inferred  # Expected VAF
    
    # Compute true trajectory (from full simulation, subsampled to observed points)
    t_true_full, x_true_full, n_true_full, _ = run_model_new(s_true, x0_true, B, E, cfg_r)
    # Interpolate true x to observation times
    x_true_obs = np.interp(t_obs, t_true_full, x_true_full)
    n_true_obs = np.interp(t_obs, t_true_full, n_true_full)
    p_true_obs = x_true_obs / (2.0 * (x_true_obs + n_true_obs))
    vaf_true_obs = p_true_obs
    
    # Calculate accuracy metrics
    err = s_hat - s_true
    rel_err = err / s_true if s_true != 0 else np.nan
    abs_err = abs(err)
    sq_err = err ** 2
    
    # R-squared for trajectory fit
    ss_res = np.sum((vaf_obs - vaf_inferred) ** 2)
    ss_tot = np.sum((vaf_obs - np.mean(vaf_obs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {
        "status": "ok",
        "E": float(E),
        "B": float(B),
        "s_true": float(s_true),
        "x0_true": float(x0_true),
        "rep": int(rep),
        "n_points": int(len(t_obs)),
        "t_start": float(t_obs[0]),
        "t_end": float(t_obs[-1]),
        "s_hat": float(s_hat),
        "x0_hat": float(x0_hat),
        "err": float(err),
        "rel_err": float(rel_err),
        "abs_err": float(abs_err),
        "sq_err": float(sq_err),
        "r_squared": float(r_squared),
        "nll": float(opt_result.fun),
        "convergence": int(opt_result.success),
        # Store trajectories for plotting
        "t_obs": t_obs,
        "vaf_obs": vaf_obs,
        "vaf_inferred": vaf_inferred,
        "vaf_true": vaf_true_obs,
        "dp_obs": dp_obs,
        "ao_obs": ao_obs,
    }


def plot_validation_trajectory(result: dict, out_dir: Path, show_true: bool = True):
    """
    Plot observed vs inferred trajectory for a single validation case.
    """
    if result.get("status") != "ok":
        return None
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                            gridspec_kw={'height_ratios': [3, 1]})
    
    t = result["t_obs"]
    vaf_obs = result["vaf_obs"]
    vaf_inf = result["vaf_inferred"]
    vaf_true = result["vaf_true"]
    dp = result["dp_obs"]
    ao = result["ao_obs"]
    
    # Main trajectory plot
    ax1 = axes[0]
    
    # Plot true trajectory if requested
    if show_true:
        ax1.plot(t, vaf_true, 'g-', linewidth=2, alpha=0.7, label='True VAF (model)')
    
    # Plot inferred trajectory
    ax1.plot(t, vaf_inf, 'r--', linewidth=2, label=f'Inferred (s={result["s_hat"]:.4f})')
    
    # Plot observations with error bars (approximate 95% CI for binomial)
    p_obs = ao / dp
    se = np.sqrt(p_obs * (1 - p_obs) / dp)
    ax1.errorbar(t, vaf_obs, yerr=1.96*se, fmt='bo', alpha=0.6, 
                markersize=6, capsize=3, label='Observed VAF ±95% CI')
    
    ax1.axhline(0.05, color='gray', linestyle=':', alpha=0.5, label='Detection threshold')
    ax1.set_ylabel('Variant Allele Frequency (VAF)', fontsize=12)
    ax1.set_title(
        f'Validation: E={result["E"]:.0e}, B={result["B"]:.4f}, '
        f's_true={result["s_true"]:.3f}\n'
        f'Inferred s={result["s_hat"]:.4f} (error={result["err"]:+.4f}, '
        f'R²={result["r_squared"]:.3f})',
        fontsize=11
    )
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(max(vaf_obs)*1.2, 0.1))
    
    # Residuals plot
    ax2 = axes[1]
    residuals = vaf_obs - vaf_inf
    ax2.bar(t, residuals, width=np.diff(t).mean()*0.8 if len(t) > 1 else 2, 
           color='purple', alpha=0.6, label='Residuals (obs - inferred)')
    ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_path = out_dir / f"validation_E{result['E']:.0e}_B{result['B']:.4f}_rep{result['rep']}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved trajectory: {out_path}")
    return out_path


def plot_accuracy_summary(results: list[dict], out_dir: Path, s_true: float):
    """
    Create summary visualization showing accuracy across replicates for specific E,B.
    """
    ok_results = [r for r in results if r.get("status") == "ok"]
    if not ok_results:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    s_hats = [r["s_hat"] for r in ok_results]
    errors = [r["err"] for r in ok_results]
    r2s = [r["r_squared"] for r in ok_results]
    
    # 1. Distribution of inferred s
    ax1 = axes[0, 0]
    ax1.hist(s_hats, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(s_true, color='red', linestyle='--', linewidth=2, label=f'True s={s_true}')
    ax1.axvline(np.mean(s_hats), color='blue', linestyle='-', linewidth=2, 
               label=f'Mean inferred={np.mean(s_hats):.4f}')
    ax1.set_xlabel('Inferred fitness (s)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of Inferred Fitness', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error distribution
    ax2 = axes[0, 1]
    ax2.hist(errors, bins=15, color='salmon', edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='black', linestyle='--', linewidth=2)
    ax2.axvline(np.mean(errors), color='red', linestyle='-', linewidth=2,
               label=f'Mean bias={np.mean(errors):+.4f}')
    ax2.set_xlabel('Error (inferred - true)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Error Distribution', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Inferred vs True (scatter with identity line)
    ax3 = axes[1, 0]
    # Add jitter for visibility if needed
    x_vals = [s_true] * len(s_hats)
    jitter = np.random.normal(0, 0.005, len(s_hats))
    ax3.scatter(np.array(x_vals) + jitter, s_hats, alpha=0.6, s=100, color='green')
    ax3.plot([s_true-0.1, s_true+0.1], [s_true-0.1, s_true+0.1], 
            'k--', label='Perfect inference')
    ax3.set_xlabel('True fitness', fontsize=11)
    ax3.set_ylabel('Inferred fitness', fontsize=11)
    ax3.set_title('Inferred vs True Fitness', fontsize=12)
    ax3.set_xlim(s_true-0.05, s_true+0.05)
    ax3.set_ylim(s_true-0.05, s_true+0.05)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. R-squared distribution
    ax4 = axes[1, 1]
    ax4.hist(r2s, bins=15, color='lightgreen', edgecolor='black', alpha=0.7)
    ax4.axvline(np.mean(r2s), color='red', linestyle='-', linewidth=2,
               label=f'Mean R²={np.mean(r2s):.3f}')
    ax4.set_xlabel('R² (trajectory fit)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Trajectory Fit Quality', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add summary text
    fig.suptitle(
        f'Accuracy Summary: E={ok_results[0]["E"]:.0e}, B={ok_results[0]["B"]:.4f}, '
        f's_true={s_true}\n'
        f'RMSE={np.sqrt(np.mean([e**2 for e in errors])):.4f}, '
        f'MAE={np.mean(np.abs(errors)):.4f}, '
        f'Bias={np.mean(errors):+.4f}',
        fontsize=13, y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_path = out_dir / "accuracy_summary.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved summary: {out_path}")
    return out_path


def create_accuracy_table(results: list[dict], out_path: Path):
    """
    Create a formatted text/CSV table of accuracy metrics.
    """
    ok_results = [r for r in results if r.get("status") == "ok"]
    if not ok_results:
        return None
    
    # Compute statistics
    s_hats = [r["s_hat"] for r in ok_results]
    errors = [r["err"] for r in ok_results]
    
    stats = {
        "E": ok_results[0]["E"],
        "B": ok_results[0]["B"],
        "s_true": ok_results[0]["s_true"],
        "n_replicates": len(ok_results),
        "s_mean": np.mean(s_hats),
        "s_std": np.std(s_hats),
        "s_median": np.median(s_hats),
        "bias": np.mean(errors),
        "mae": np.mean(np.abs(errors)),
        "rmse": np.sqrt(np.mean([e**2 for e in errors])),
        "min_err": min(errors),
        "max_err": max(errors),
        "mean_r2": np.mean([r["r_squared"] for r in ok_results]),
        "convergence_rate": np.mean([r["convergence"] for r in ok_results]),
    }
    
    # Write formatted table
    with open(out_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ACCURACY VALIDATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  E (carrying capacity):  {stats['E']:.0e}\n")
        f.write(f"  B (competition):        {stats['B']:.6f}\n")
        f.write(f"  True fitness (s):       {stats['s_true']:.4f}\n\n")
        f.write(f"Replication:              {stats['n_replicates']} runs\n\n")
        f.write("-" * 60 + "\n")
        f.write("INFERENCE ACCURACY METRICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Mean inferred s:        {stats['s_mean']:.6f} ± {stats['s_std']:.6f}\n")
        f.write(f"Median inferred s:      {stats['s_median']:.6f}\n")
        f.write(f"Bias (mean error):      {stats['bias']:+.6f}\n")
        f.write(f"Mean Absolute Error:    {stats['mae']:.6f}\n")
        f.write(f"Root Mean Square Error: {stats['rmse']:.6f}\n")
        f.write(f"Error range:            [{stats['min_err']:+.6f}, {stats['max_err']:+.6f}]\n")
        f.write(f"Mean R² (trajectory):   {stats['mean_r2']:.4f}\n")
        f.write(f"Optimizer convergence:  {stats['convergence_rate']*100:.1f}%\n")
        f.write("=" * 60 + "\n")
        
        # Individual results
        f.write("\nIndividual Replicate Results:\n")
        f.write(f"{'Rep':>4} | {'s_hat':>10} | {'Error':>10} | {'R²':>8} | {'Conv':>4}\n")
        f.write("-" * 50 + "\n")
        for r in ok_results:
            f.write(f"{r['rep']:4d} | {r['s_hat']:10.6f} | {r['err']:+10.6f} | "
                   f"{r['r_squared']:8.4f} | {r['convergence']:4d}\n")
    
    print(f"Saved accuracy table: {out_path}")
    return stats


def run_validation_mode(
    out_dir: Path,
    E: float,
    B: float,
    s_true: float,
    x0_true: float,
    reps: int,
    cfg: SimConfig,
    s_upper: float = 1.0,
    map_starts: int = 6,
):
    """
    Run detailed validation for a specific (E, B) combination.
    Generates trajectory plots, summary statistics, and accuracy table.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"VALIDATION MODE: E={E:.0e}, B={B:.6f}, s_true={s_true}")
    print(f"{'='*60}")
    
    results = []
    for r in range(reps):
        print(f"  Running replicate {r+1}/{reps}...", end=" ")
        res = run_single_validation(
            E=E, B=B, s_true=s_true, x0_true=x0_true,
            cfg=cfg, s_upper=s_upper, map_starts=map_starts, rep=r
        )
        results.append(res)
        if res["status"] == "ok":
            print(f"s_hat={res['s_hat']:.4f}, err={res['err']:+.4f}")
            # Plot individual trajectory
            plot_validation_trajectory(res, out_dir)
        else:
            print(f"FAILED: {res['status']}")
    
    # Summary analysis
    ok_results = [r for r in results if r.get("status") == "ok"]
    n_ok = len(ok_results)
    print(f"\nSuccessful runs: {n_ok}/{reps}")
    
    if n_ok == 0:
        print("No successful runs to analyze!")
        return results, None
    
    # Generate summary plots
    plot_accuracy_summary(ok_results, out_dir, s_true)
    
    # Create accuracy table
    table_path = out_dir / "accuracy_table.txt"
    stats = create_accuracy_table(ok_results, table_path)
    
    # Save detailed results to JSON
    json_path = out_dir / "validation_results.json"
    # Convert numpy arrays to lists for JSON serialization
    json_results = []
    for r in results:
        r_copy = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                 for k, v in r.items()}
        json_results.append(r_copy)
    
    with open(json_path, 'w') as f:
        json.dump({
            "parameters": {"E": E, "B": B, "s_true": s_true, "x0_true": x0_true, "reps": reps},
            "summary": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in stats.items()},
            "replicates": json_results
        }, f, indent=2)
    
    print(f"\nValidation complete. Results saved to: {out_dir}")
    return results, stats


# -----------------------------
# E-B Sweep Implementation (Original + Log Scale Support)
# -----------------------------

def run_E_B_sweep(
    out_dir: Path,
    E_values: list[float],
    B_values: list[float],
    s_true: float,
    x0_true: float,
    reps: int,
    cfg: SimConfig,
    s_upper: float = 1.0,
    map_starts: int = 6,
    progress_every: int = 10,
):
    """
    Run sweep over E and B with fixed true fitness s.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define all fieldnames upfront
    fieldnames = [
        "E", "B", "fitness_true", "x0_true", "rep", "n_points",
        "t_start", "t_end", "vaf_start", "vaf_end",
        "status", "fitness_map", "x0_map", "err", "abs_err", "sq_err"
    ]
    
    rows: list[dict] = []
    total = int(reps) * len(E_values) * len(B_values)
    run_i = 0
    base_seed = int(cfg.seed)
    
    for E in E_values:
        for B in B_values:
            for r in range(int(reps)):
                run_i += 1
                cfg_r = SimConfig(**{**asdict(cfg), "seed": base_seed + run_i})
                
                if progress_every and (run_i % int(progress_every) == 0):
                    print(f"[sweep] {run_i}/{total} E={E:g} B={B:g} s={s_true:g} (rep {r})", flush=True)
                
                # Initialize row with defaults
                row = {
                    "E": float(E),
                    "B": float(B),
                    "fitness_true": float(s_true),
                    "x0_true": float(x0_true),
                    "rep": int(r),
                    "n_points": 0,
                    "t_start": np.nan,
                    "t_end": np.nan,
                    "vaf_start": np.nan,
                    "vaf_end": np.nan,
                    "status": "pending",
                    "fitness_map": np.nan,
                    "x0_map": np.nan,
                    "err": np.nan,
                    "abs_err": np.nan,
                    "sq_err": np.nan,
                }
                
                try:
                    t, vaf, dp, ao = generate_synth_new(
                        fitness=s_true, x0=x0_true, E=E, B=B, cfg=cfg_r
                    )
                except Exception as e:
                    row["status"] = f"sim_failed:{type(e).__name__}"
                    rows.append(row)
                    continue
                
                masked = apply_threshold_and_window(t, vaf, dp, ao, cfg_r)
                if masked is None:
                    row["status"] = "skipped_threshold/window"
                    rows.append(row)
                    continue
                
                t2, v2, dp2, ao2 = masked
                row["n_points"] = int(len(t2))
                row["t_start"] = float(t2[0])
                row["t_end"] = float(t2[-1])
                row["vaf_start"] = float(v2[0])
                row["vaf_end"] = float(v2[-1])
                
                try:
                    s_hat, x0_hat, _ = fit_simple_exponential_map(
                        t2, dp2, ao2,
                        K=float(cfg_r.K),
                        s_upper=float(s_upper),
                        n_starts=int(map_starts),
                        seed=int(cfg_r.seed),
                    )
                    row["status"] = "ok"
                    row["fitness_map"] = float(s_hat)
                    row["x0_map"] = float(x0_hat)
                    row["err"] = float(s_hat - s_true)
                    row["abs_err"] = float(abs(s_hat - s_true))
                    row["sq_err"] = float((s_hat - s_true) ** 2)
                except Exception as e:
                    row["status"] = f"fit_failed:{type(e).__name__}"
                
                rows.append(row)
    
    csv_path = out_dir / "sweep_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    
    return rows, csv_path


def _load_results(csv_path: Path) -> list[dict]:
    """Reload rows from CSV."""
    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


# -----------------------------
# Visualization Functions (with Log Scale Support)
# -----------------------------

def _format_B_labels(B_values: list[float], log_scale: bool = False) -> list[str]:
    """Format B values for labels, using scientific notation for small values."""
    if log_scale:
        return [f"{B:.0e}" if B < 0.01 else f"{B:.2f}" for B in B_values]
    else:
        return [f"{B:.4f}" if B < 0.01 else f"{B:.2f}" for B in B_values]


def plot_accuracy_heatmap(rows: list[dict], out_dir: Path, s_true: float, metric: str = "rmse", log_B: bool = False):
    """
    Create heatmap of inference accuracy across E and B.
    
    Parameters:
    -----------
    rows : simulation results
    out_dir : output directory
    s_true : true fitness value (for reference line)
    metric : 'rmse', 'bias', 'mae', or 'fraction_within_tol'
    log_B : use log scale for B axis
    """
    ok = [r for r in rows if r.get("status") == "ok" and np.isfinite(r.get("fitness_map", np.nan))]
    if not ok:
        print("No successful runs to plot")
        return None
    
    E_vals = sorted({r["E"] for r in ok})
    B_vals = sorted({r["B"] for r in ok})
    
    # Create grid
    grid = np.full((len(E_vals), len(B_vals)), np.nan, dtype=float)
    
    for iE, E in enumerate(E_vals):
        for jB, B in enumerate(B_vals):
            subset = [r for r in ok if r["E"] == E and r["B"] == B]
            if not subset:
                continue
            
            errs = np.array([r["err"] for r in subset], dtype=float)
            
            if metric == "rmse":
                grid[iE, jB] = float(np.sqrt(np.mean(errs ** 2)))
            elif metric == "bias":
                grid[iE, jB] = float(np.mean(errs))
            elif metric == "mae":
                grid[iE, jB] = float(np.mean(np.abs(errs)))
            elif metric == "fraction_within_tol":
                # Requires tolerance parameter - default 0.05
                tol = 0.05
                grid[iE, jB] = float(np.mean(np.abs(errs) <= tol))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if metric == "fraction_within_tol":
        cmap = "RdYlGn"
        vmin, vmax = 0, 1
        title_suffix = f"Fraction within ±0.05 of true s={s_true}"
        cbar_label = "Fraction accurate"
    else:
        cmap = "viridis_r" if metric in ["rmse", "mae"] else "coolwarm"
        vmin, vmax = None, None
        title_suffix = f"{metric.upper()} of inferred fitness (true s={s_true})"
        cbar_label = metric.upper()
    
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, 
                   interpolation="nearest", origin="lower")
    
    ax.set_yticks(np.arange(len(E_vals)))
    ax.set_yticklabels([f"{E:.0e}" for E in E_vals])
    ax.set_xticks(np.arange(len(B_vals)))
    
    # Use formatted labels
    B_labels = _format_B_labels(B_vals, log_scale=log_B)
    ax.set_xticklabels(B_labels, rotation=45, ha="right")
    
    ax.set_xlabel("B (competition parameter)", fontsize=12)
    ax.set_ylabel("E (carrying capacity)", fontsize=12)
    ax.set_title(f"Fitness Inference Accuracy\n{title_suffix}", fontsize=13)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=11)
    
    # Add text annotations
    for i in range(len(E_vals)):
        for j in range(len(B_vals)):
            if not np.isnan(grid[i, j]):
                text_color = "white" if grid[i, j] < (0.5 if metric == "fraction_within_tol" else np.nanmax(grid)/2) else "black"
                text = ax.text(j, i, f"{grid[i, j]:.3f}",
                             ha="center", va="center", color=text_color,
                             fontsize=8)
    
    plt.tight_layout()
    out_path = out_dir / f"heatmap_{metric}{'_logB' if log_B else ''}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_bias_vs_E_and_B(rows: list[dict], out_dir: Path, s_true: float, log_B: bool = False):
    """Plot bias (mean error) as function of E and B."""
    ok = [r for r in rows if r.get("status") == "ok" and np.isfinite(r.get("fitness_map", np.nan))]
    if not ok:
        return None
    
    E_vals = sorted({r["E"] for r in ok})
    B_vals = sorted({r["B"] for r in ok})
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Bias vs B for different E
    ax1 = axes[0]
    for E in E_vals:
        biases = []
        for B in B_vals:
            subset = [r for r in ok if r["E"] == E and r["B"] == B]
            if subset:
                errs = [r["err"] for r in subset]
                biases.append(np.mean(errs))
            else:
                biases.append(np.nan)
        ax1.plot(B_vals, biases, marker="o", linewidth=2, label=f"E={E:.0e}")
    
    ax1.axhline(0, color="black", linestyle="--", alpha=0.5)
    if log_B and min(B_vals) > 0:
        ax1.set_xscale('log')
    ax1.set_xlabel("B (log scale)" if log_B else "B", fontsize=12)
    ax1.set_ylabel("Bias (mean error)", fontsize=12)
    ax1.set_title(f"Bias vs B (true s={s_true})", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both' if log_B else 'major')
    
    # Right: Bias vs E for different B
    ax2 = axes[1]
    for B in B_vals:
        biases = []
        for E in E_vals:
            subset = [r for r in ok if r["E"] == E and r["B"] == B]
            if subset:
                errs = [r["err"] for r in subset]
                biases.append(np.mean(errs))
            else:
                biases.append(np.nan)
        # Format B label for legend
        B_label = f"{B:.0e}" if B < 0.01 else f"{B:.2f}"
        ax2.plot(E_vals, biases, marker="s", linewidth=2, label=f"B={B_label}")
    
    ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax2.set_xlabel("E", fontsize=12)
    ax2.set_ylabel("Bias (mean error)", fontsize=12)
    ax2.set_title(f"Bias vs E (true s={s_true})", fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = out_dir / f"bias_vs_E_and_B{'_logB' if log_B else ''}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_rmse_vs_E_and_B(rows: list[dict], out_dir: Path, s_true: float, log_B: bool = False):
    """Plot RMSE as function of E and B."""
    ok = [r for r in rows if r.get("status") == "ok" and np.isfinite(r.get("fitness_map", np.nan))]
    if not ok:
        return None
    
    E_vals = sorted({r["E"] for r in ok})
    B_vals = sorted({r["B"] for r in ok})
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: RMSE vs B for different E
    ax1 = axes[0]
    for E in E_vals:
        rmses = []
        for B in B_vals:
            subset = [r for r in ok if r["E"] == E and r["B"] == B]
            if subset:
                errs = [r["err"] for r in subset]
                rmses.append(np.sqrt(np.mean(np.array(errs) ** 2)))
            else:
                rmses.append(np.nan)
        ax1.plot(B_vals, rmses, marker="o", linewidth=2, label=f"E={E:.0e}")
    
    if log_B and min(B_vals) > 0:
        ax1.set_xscale('log')
    ax1.set_xlabel("B (log scale)" if log_B else "B", fontsize=12)
    ax1.set_ylabel("RMSE", fontsize=12)
    ax1.set_title(f"RMSE vs B (true s={s_true})", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both' if log_B else 'major')
    
    # Right: RMSE vs E for different B
    ax2 = axes[1]
    for B in B_vals:
        rmses = []
        for E in E_vals:
            subset = [r for r in ok if r["E"] == E and r["B"] == B]
            if subset:
                errs = [r["err"] for r in subset]
                rmses.append(np.sqrt(np.mean(np.array(errs) ** 2)))
            else:
                rmses.append(np.nan)
        B_label = f"{B:.0e}" if B < 0.01 else f"{B:.2f}"
        ax2.plot(E_vals, rmses, marker="s", linewidth=2, label=f"B={B_label}")
    
    ax2.set_xlabel("E", fontsize=12)
    ax2.set_ylabel("RMSE", fontsize=12)
    ax2.set_title(f"RMSE vs E (true s={s_true})", fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = out_dir / f"rmse_vs_E_and_B{'_logB' if log_B else ''}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_distribution_by_EB(rows: list[dict], out_dir: Path, s_true: float, log_B: bool = False):
    """Box plots showing distribution of inferred fitness across E and B."""
    ok = [r for r in rows if r.get("status") == "ok" and np.isfinite(r.get("fitness_map", np.nan))]
    if not ok:
        return None
    
    E_vals = sorted({r["E"] for r in ok})
    B_vals = sorted({r["B"] for r in ok})
    
    fig, axes = plt.subplots(len(E_vals), 1, figsize=(12, 4 * len(E_vals)), sharex=True)
    if len(E_vals) == 1:
        axes = [axes]
    
    for idx, E in enumerate(E_vals):
        ax = axes[idx]
        data = []
        labels = []
        for B in B_vals:
            subset = [r["fitness_map"] for r in ok if r["E"] == E and r["B"] == B]
            if subset:
                data.append(subset)
                # Format label based on magnitude
                B_label = f"{B:.0e}" if B < 0.01 else f"{B:.2f}"
                labels.append(f"B={B_label}")
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            ax.axhline(s_true, color="red", linestyle="--", linewidth=2, label=f"True s={s_true}")
            ax.set_ylabel("Inferred fitness", fontsize=11)
            ax.set_title(f"E = {E:.0e}", fontsize=12)
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)
            
            # Use log scale for x-axis if requested and B spans multiple orders
            if log_B and max(B_vals) / min(B_vals) > 10:
                ax.set_xscale('log')
                # Re-set tick labels since boxplot doesn't auto-handle log well
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, rotation=45, ha="right")
    
    axes[-1].set_xlabel("B values", fontsize=12)
    fig.suptitle("Distribution of Inferred Fitness by E and B", fontsize=14, y=1.02)
    plt.tight_layout()
    
    out_path = out_dir / f"distribution_by_EB{'_logB' if log_B else ''}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_scatter_facet(rows: list[dict], out_dir: Path, s_true: float, log_B: bool = False):
    """Facet scatter plot: inferred vs nothing (just show spread), faceted by E and B."""
    ok = [r for r in rows if r.get("status") == "ok" and np.isfinite(r.get("fitness_map", np.nan))]
    if not ok:
        return None
    
    E_vals = sorted({r["E"] for r in ok})
    B_vals = sorted({r["B"] for r in ok})
    
    nE, nB = len(E_vals), len(B_vals)
    fig, axes = plt.subplots(nE, nB, figsize=(4 * nB, 3 * nE), sharex=True, sharey=True)
    
    if nE == 1 and nB == 1:
        axes = [[axes]]
    elif nE == 1:
        axes = [axes]
    elif nB == 1:
        axes = [[ax] for ax in axes]
    
    for i, E in enumerate(E_vals):
        for j, B in enumerate(B_vals):
            ax = axes[i][j]
            subset = [r["fitness_map"] for r in ok if r["E"] == E and r["B"] == B]
            
            if subset:
                y = np.array(subset)
                x = np.random.normal(0, 0.1, size=len(y))  # jitter
                ax.scatter(x, y, alpha=0.6, s=50)
                ax.axhline(s_true, color="red", linestyle="--", alpha=0.7)
                
                # Add statistics
                mean_s = np.mean(y)
                rmse = np.sqrt(np.mean((y - s_true) ** 2))
                ax.text(0.5, 0.95, f"mean={mean_s:.3f}\nrmse={rmse:.3f}",
                       transform=ax.transAxes, ha="center", va="top",
                       bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            
            B_label = f"{B:.0e}" if B < 0.01 else f"{B:.2f}"
            ax.set_title(f"E={E:.0e}, B={B_label}", fontsize=10)
            ax.set_ylim(s_true - 0.5, s_true + 0.5)
            ax.grid(True, alpha=0.3)
    
    # Add common labels
    for ax in axes[-1]:
        ax.set_xlabel("Jittered samples")
    for ax in [row[0] for row in axes]:
        ax.set_ylabel("Inferred fitness")
    
    fig.suptitle(f"Inferred Fitness Distribution by (E, B)\nTrue s={s_true}", fontsize=14)
    plt.tight_layout()
    
    out_path = out_dir / f"scatter_facet_EB{'_logB' if log_B else ''}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


# NEW: Log-scale specific visualization for B-sensitivity
def plot_B_sensitivity_log(rows: list[dict], out_dir: Path, s_true: float):
    """
    Specialized plot for B sensitivity with log-scaled B axis.
    Shows how inference accuracy changes across orders of magnitude of B.
    """
    ok = [r for r in rows if r.get("status") == "ok" and np.isfinite(r.get("fitness_map", np.nan))]
    if not ok:
        return None
    
    E_vals = sorted({r["E"] for r in ok})
    B_vals = sorted({r["B"] for r in ok})
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. RMSE vs B (log scale) - all E combined and per-E
    ax1 = axes[0, 0]
    for E in E_vals:
        rmses = []
        for B in B_vals:
            subset = [r for r in ok if r["E"] == E and r["B"] == B]
            if subset:
                errs = [r["err"] for r in subset]
                rmses.append(np.sqrt(np.mean(np.array(errs) ** 2)))
            else:
                rmses.append(np.nan)
        ax1.plot(B_vals, rmses, marker="o", linewidth=2, label=f"E={E:.0e}")
    
    ax1.set_xscale('log')
    ax1.set_xlabel("B (competition parameter, log scale)", fontsize=11)
    ax1.set_ylabel("RMSE", fontsize=11)
    ax1.set_title("RMSE vs B (log scale)", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, which='both')
    
    # 2. Bias vs B (log scale)
    ax2 = axes[0, 1]
    for E in E_vals:
        biases = []
        for B in B_vals:
            subset = [r for r in ok if r["E"] == E and r["B"] == B]
            if subset:
                errs = [r["err"] for r in subset]
                biases.append(np.mean(errs))
            else:
                biases.append(np.nan)
        ax2.plot(B_vals, biases, marker="s", linewidth=2, label=f"E={E:.0e}")
    
    ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax2.set_xscale('log')
    ax2.set_xlabel("B (log scale)", fontsize=11)
    ax2.set_ylabel("Bias (mean error)", fontsize=11)
    ax2.set_title("Bias vs B (log scale)", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')
    
    # 3. Fraction within tolerance vs B (log scale)
    ax3 = axes[1, 0]
    tol = 0.05
    for E in E_vals:
        frac_accurate = []
        for B in B_vals:
            subset = [r for r in ok if r["E"] == E and r["B"] == B]
            if subset:
                errs = [r["err"] for r in subset]
                frac_accurate.append(np.mean(np.abs(errs) <= tol))
            else:
                frac_accurate.append(np.nan)
        ax3.plot(B_vals, frac_accurate, marker="^", linewidth=2, label=f"E={E:.0e}")
    
    ax3.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="Perfect")
    ax3.set_xscale('log')
    ax3.set_xlabel("B (log scale)", fontsize=11)
    ax3.set_ylabel(f"Fraction within ±{tol}", fontsize=11)
    ax3.set_title("Inference Accuracy vs B", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_ylim(0, 1.05)
    
    # 4. MAE vs B (log scale) with trend annotation
    ax4 = axes[1, 1]
    for E in E_vals:
        maes = []
        for B in B_vals:
            subset = [r for r in ok if r["E"] == E and r["B"] == B]
            if subset:
                errs = [r["err"] for r in subset]
                maes.append(np.mean(np.abs(errs)))
            else:
                maes.append(np.nan)
        ax4.plot(B_vals, maes, marker="D", linewidth=2, label=f"E={E:.0e}")
    
    ax4.set_xscale('log')
    ax4.set_xlabel("B (log scale)", fontsize=11)
    ax4.set_ylabel("MAE", fontsize=11)
    ax4.set_title("Mean Absolute Error vs B", fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    
    fig.suptitle(f"B-Sensitivity Analysis (Log Scale)\nTrue s={s_true}", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_path = out_dir / "B_sensitivity_log.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Sweep E and B parameters to analyze fitness inference accuracy."
    )
    ap.add_argument("--out", default="exports/EB_sweep", help="Output directory")
    ap.add_argument("--seed", type=int, default=123)
    
    # Simulation parameters
    ap.add_argument("--K", type=float, default=1e5)
    ap.add_argument("--t_end", type=float, default=200.0)
    ap.add_argument("--t_points", type=int, default=2000)
    ap.add_argument("--dp_mean", type=float, default=5000.0)
    ap.add_argument("--dp_sd", type=float, default=2000.0)
    ap.add_argument("--dp_min", type=int, default=100)
    ap.add_argument("--sample_every", type=int, default=60)
    ap.add_argument("--vaf_threshold", type=float, default=0.05)
    ap.add_argument("--followup_window", type=float, default=80.0)
    
    # Sweep parameters
    ap.add_argument("--E", default="100000,200000,500000,1000000", 
                  help="Comma-separated E values (carrying capacity)")
    ap.add_argument("--B", default="0.0001,0.001,0.01,0.1,1.0,10.0", 
                  help="Comma-separated B values (supports log range like 0.0001,0.001,0.01,0.1,1,10.0)")
    ap.add_argument("--s_true", type=float, default=0.3, 
                  help="True fitness value (fixed)")
    ap.add_argument("--x0_true", type=float, default=1.0, 
                  help="True initial clone size")
    ap.add_argument("--reps", type=int, default=20, 
                  help="Replicates per (E,B) combination")
    
    # Inference parameters
    ap.add_argument("--s_upper", type=float, default=1.0)
    ap.add_argument("--map_starts", type=int, default=6)
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--no_plots", action="store_true")
    
    # Plotting options
    ap.add_argument("--log_B", action="store_true",
                   help="Use log scale for B in plots")
    
    # Validation mode
    ap.add_argument("--validate", action="store_true",
                   help="Run detailed validation for specific E,B values")
    ap.add_argument("--validate_E", type=float, default=None,
                   help="E value for validation mode (required if --validate)")
    ap.add_argument("--validate_B", type=float, default=None,
                   help="B value for validation mode (required if --validate)")
    ap.add_argument("--validate_reps", type=int, default=50,
                   help="Replicates for validation mode")
    
    args = ap.parse_args()
    
    def parse_list(spec: str):
        return [float(x.strip()) for x in spec.split(",") if x.strip() != ""]
    
    cfg = SimConfig(
        K=float(args.K),
        t_end=float(args.t_end),
        t_points=int(args.t_points),
        dp_mean=float(args.dp_mean),
        dp_sd=float(args.dp_sd),
        dp_min=int(args.dp_min),
        sample_every=int(args.sample_every),
        vaf_threshold=float(args.vaf_threshold),
        followup_window=float(args.followup_window),
        seed=int(args.seed),
    )
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = {
        "sim_config": asdict(cfg),
        "sweep": {
            "E_values": parse_list(args.E),
            "B_values": parse_list(args.B),
            "s_true": float(args.s_true),
            "x0_true": float(args.x0_true),
            "reps": int(args.reps),
            "s_upper": float(args.s_upper),
            "map_starts": int(args.map_starts),
        }
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    
    # VALIDATION MODE
    if args.validate:
        if args.validate_E is None or args.validate_B is None:
            print("ERROR: --validate requires both --validate_E and --validate_B")
            return 1
        
        val_dir = out_dir / f"validation_E{args.validate_E:.0e}_B{args.validate_B:.6f}"
        results, stats = run_validation_mode(
            out_dir=val_dir,
            E=float(args.validate_E),
            B=float(args.validate_B),
            s_true=float(args.s_true),
            x0_true=float(args.x0_true),
            reps=int(args.validate_reps),
            cfg=cfg,
            s_upper=float(args.s_upper),
            map_starts=int(args.map_starts),
        )
        
        if stats:
            print(f"\n{'='*60}")
            print("VALIDATION SUMMARY")
            print(f"{'='*60}")
            print(f"Parameters: E={stats['E']:.0e}, B={stats['B']:.6f}, s_true={stats['s_true']:.4f}")
            print(f"Replicates:  {stats['n_replicates']}")
            print(f"Mean inferred s: {stats['s_mean']:.6f} ± {stats['s_std']:.6f}")
            print(f"Bias:        {stats['bias']:+.6f}")
            print(f"RMSE:        {stats['rmse']:.6f}")
            print(f"MAE:         {stats['mae']:.6f}")
            print(f"Mean R²:     {stats['mean_r2']:.4f}")
            print(f"{'='*60}")
        return 0
    
    # STANDARD SWEEP MODE
    print(f"Starting E-B sweep with s_true={args.s_true}")
    print(f"E values: {parse_list(args.E)}")
    print(f"B values: {parse_list(args.B)}")
    print(f"Replicates per combination: {args.reps}")
    if args.log_B:
        print("Using log scale for B in plots")
    
    rows, csv_path = run_E_B_sweep(
        out_dir=out_dir,
        E_values=parse_list(args.E),
        B_values=parse_list(args.B),
        s_true=float(args.s_true),
        x0_true=float(args.x0_true),
        reps=int(args.reps),
        cfg=cfg,
        s_upper=float(args.s_upper),
        map_starts=int(args.map_starts),
        progress_every=int(args.progress_every),
    )
    
    print(f"\nResults saved to: {csv_path}")
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        rows = _load_results(csv_path)
        
        log_B = args.log_B
        
        # Standard plots with optional log B
        plot_accuracy_heatmap(rows, out_dir, float(args.s_true), metric="rmse", log_B=log_B)
        plot_accuracy_heatmap(rows, out_dir, float(args.s_true), metric="bias", log_B=log_B)
        plot_accuracy_heatmap(rows, out_dir, float(args.s_true), metric="mae", log_B=log_B)
        plot_accuracy_heatmap(rows, out_dir, float(args.s_true), metric="fraction_within_tol", log_B=log_B)
        
        plot_bias_vs_E_and_B(rows, out_dir, float(args.s_true), log_B=log_B)
        plot_rmse_vs_E_and_B(rows, out_dir, float(args.s_true), log_B=log_B)
        plot_distribution_by_EB(rows, out_dir, float(args.s_true), log_B=log_B)
        plot_scatter_facet(rows, out_dir, float(args.s_true), log_B=log_B)
        
        # NEW: Specialized log-scale B sensitivity plot
        if log_B or max(parse_list(args.B)) / min(parse_list(args.B)) > 10:
            plot_B_sensitivity_log(rows, out_dir, float(args.s_true))
        
        print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()

